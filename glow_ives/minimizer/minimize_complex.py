import sys, time, os, io
from simtk import unit, openmm
from openmm import app as openmm_app, Platform, LangevinIntegrator
from openmm.app import PDBFile, Simulation, Modeller
from openmm.app.internal.pdbstructure import PdbStructure
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SystemGenerator

import click
import tempfile
from copy import deepcopy
from rdkit import Chem

import glow_ives.minimizer.cleanup as cleanup
from glow_ives.minimizer.renumber_pdb import renumber_pdb
from glow_ives.utils.utils import run_cmd


def _get_pdb_string(topology: openmm_app.Topology, positions: unit.Quantity):
    """Returns a pdb string provided OpenMM topology and positions."""
    with io.StringIO() as f:
        openmm_app.PDBFile.writeFile(topology, positions, f)
        return f.getvalue()


def clean_protein(pdb_file, checks: bool = False):
    # Clean pdb.
    alterations_info = {}
    fixed_pdb = cleanup.fix_pdb(pdb_file, alterations_info)
    fixed_pdb_file = io.StringIO(fixed_pdb)
    pdb_structure = PdbStructure(fixed_pdb_file)
    cleanup.clean_structure(pdb_structure, alterations_info)
    print("alterations info: %s", alterations_info)
    # Write pdb file of cleaned structure.
    as_file = PDBFile(pdb_structure)
    pdb_string = _get_pdb_string(as_file.getTopology(), as_file.getPositions())
    return pdb_string


AA_CODES = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
     'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
     'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ',
     ]

# Function to add protein backbone and ligand heavy atom position constraints (skip binding pocket residues)
def add_backbone_ligand_pos_constraints(system, positions, atoms, stiffness, ligand_name, binding_pocket_residues):
    all_residues = set()
    all_names = set()
    all_elements = set()

    copy_sys = deepcopy(system)

    force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", stiffness)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    for i, (atom_pos, atom) in enumerate(zip(positions, atoms)):
        # Ignore hydrogens
        if atom.element == openmm_app.element.hydrogen:
            continue
        # Do not restrain binding pocket residues
        ## NOTE: binding_pocket_residues should be indexed by the openmm residue internal numbering
        if (atom.residue.index) in binding_pocket_residues:
            assert atom.residue.name == binding_pocket_residues[atom.residue.index]
            #print(f'Skip binding pocket residue {atom.residue.index}:{atom.residue.name}')
            continue
        if (atom.residue.name == ligand_name):
            # Add constrain to ligand
            copy_sys.setParticleMass(atom.index, 0*unit.amu)
            all_residues.add(atom.residue.name)
            all_names.add(atom.name)
            all_elements.add(atom.element.name)
        if ((atom.residue.name in AA_CODES) and (atom.name in ('CA', 'C', 'N'))):
            # Add constraint to protein
            copy_sys.setParticleMass(atom.index, 0*unit.amu)
            all_residues.add(atom.residue.name)
            all_names.add(atom.name)
            all_elements.add(atom.element.name)

    copy_sys.addForce(force)

    print('Add constraint/restraints to:')
    print('    residues:', all_residues)
    print('    names:', all_names)
    print('    elements:', all_elements)
    return copy_sys


def get_binding_pocket_residues(pdbfile, ligand_name, dist_cutoff):
    from biopandas.pdb import PandasPdb
    import pandas as pd
    ppdb = PandasPdb().read_pdb(pdbfile)

    ligand_df = ppdb.df['HETATM']
    ligand_df = ligand_df[(ligand_df['residue_name']==ligand_name[:3]) & (ligand_df['element_symbol']!='H')]
    lig_coords = ligand_df[['x_coord', 'y_coord', 'z_coord']].values

    protein_df = ppdb.df['ATOM']
    # Asign each residue to internal number to match openmm (starts from 0)
    ca_df = protein_df[['chain_id', 'residue_name', 'residue_number']].drop_duplicates(keep='first').reset_index(drop=True).rename_axis('index').reset_index()
    protein_df = protein_df.merge(ca_df)

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = {}
    count = 0
    for (index, residue_number, residue_name), res_df in protein_df.groupby(['index', 'residue_number', 'residue_name']):
        count += 1
        res_coords = res_df[['x_coord', 'y_coord', 'z_coord']].values
        if (residue_name in AA_CODES) and \
                (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
            pocket_residues[index] = residue_name
    print(f'Found {len(pocket_residues)}/{count} ({len(pocket_residues)*100.0/count:.2f}%) pocket res with dist cutoff {dist_cutoff}')
    return pocket_residues


def minimize_protein(pdb_in, mol_in, output_prot_min, dist_cutoff, platform='CPU'):
    RELAX_MAX_ITERATIONS = 0
    RELAX_ENERGY_TOLERANCE = 2.39
    RELAX_STIFFNESS = 10.0

    ENERGY = unit.kilocalories_per_mole
    LENGTH = unit.angstroms

    print(f'Processing {pdb_in} and {mol_in} with dist_cutoff {dist_cutoff}, write output to {output_prot_min}')

    platform = Platform.getPlatformByName(platform)
    if platform.getName() == 'CUDA' or platform.getName() == 'OpenCL':
        platform.setPropertyDefaultValue('Precision', 'mixed')
        print('Set precision for platform', platform.getName(), 'to mixed')

    # Read the molfile into RDKit, add Hs and create an openforcefield Molecule object
    print('Reading ligand')
    rdkitmol = Chem.MolFromMolFile(mol_in)
    print('Adding hydrogens')
    rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True)
    Chem.AssignStereochemistryFrom3D(rdkitmolh)
    try:
        ligand_mol = Molecule(rdkitmolh)
    except:
        print(f'Reload ligand with allow_undefined_stereo=True')
        ligand_mol = Molecule(rdkitmolh, allow_undefined_stereo=True)
    print(f'Ligand name: {ligand_mol.name}')
    ligand_charge = Chem.rdmolops.GetFormalCharge(rdkitmolh)
    print(f'Ligand formal charge: {ligand_charge}')
    smi = Chem.MolToSmiles(rdkitmolh)
    print(f'smiles: {smi}')

    # Sometimes if there is insertion code, openmm will fail, so renumber the residues first
    temp = tempfile.NamedTemporaryFile(suffix='.pdb')
    print(f'Renumber {pdb_in} to {temp.name}')
    renumber_pdb(pdb_in, temp.name, 1, True, True, False)
    pdb_in = temp.name

    print('Reading protein')
    fixed_pdb_str = clean_protein(pdb_in, checks=False)
    fixed_pdb_file = io.StringIO(fixed_pdb_str)
    protein_pdb = PDBFile(fixed_pdb_file)

    # Use Modeller to combine the protein and ligand into a complex
    print(f'Preparing complex')
    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    modeller.add(ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0].to_openmm())

    # Note: somehow the newest openff-toolkit will name ligand to UNK, the old version name it into the mol name
    #ligand_name = ligand_mol.name
    ligand_name = 'UNK'

    print('System has %d atoms' % modeller.topology.getNumAtoms())

    temp = tempfile.NamedTemporaryFile(suffix='.pdb')
    output_complex = temp.name
    print(f'Write input complex to {output_complex}')
    with open(output_complex, 'w') as outfile:
        PDBFile.writeFile(modeller.topology, modeller.positions, outfile)

    pocket_residues = get_binding_pocket_residues(output_complex, ligand_name=ligand_name, dist_cutoff=dist_cutoff)

    # Initialize a SystemGenerator using the GAFF for the ligand
    print('Preparing system')
    forcefield_kwargs = {
        # makes water molecules completely rigid, constraining both their bond lengths and angles
        'rigidWater': True,
        }
    # Use openmm_app.PME (Particle-mesh Ewald) by default
    system_generator = SystemGenerator(
        forcefields=['amber/ff14SB.xml'],
        small_molecule_forcefield='gaff-2.11',
        forcefield_kwargs=forcefield_kwargs,
        )

    try:
        t0 = time.time()
        system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
        t1 = time.time()
        print(f'Preparing system complete in {(t1-t0)/60.0:.2f} minutes')
    except:
        # Sometimes system generater will fail because rdkit complains that it
        # can't assign partial charges to the molecule if mol file passed as input,
        # but strangely if we precompute the partial charges with antechamber
        # it is fine (not sure why)
        current_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as wd:
            os.chdir(wd)
            # Precompute partial charges
            charged_mol2 = os.path.join(wd, 'charged.mol2')
            print(f'Write charged mol to {charged_mol2}')
            amber_cmd = f'antechamber -i {mol_in} -fi sdf -o {charged_mol2} -fo mol2 -pf yes -dr n -c bcc -nc {ligand_charge}'
            print(amber_cmd)
            run_cmd(amber_cmd, f'Failed to run antechamber on {mol_in}')
            assert os.path.exists(charged_mol2), f'{charged_mol2} does not exist'

            import pandas as pd
            with open(charged_mol2, 'r') as f:
                contents = f.read()
                molblock = ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]][0]
                charges_df = pd.read_csv(io.StringIO(molblock.split("@<TRIPOS>ATOM\n")[1:][0].split('@<TRIPOS>')[0]), delim_whitespace=True, header=None)
                charges = charges_df.values[:,-1]

            # Assign partial charges from a list-like object
            ligand_mol.partial_charges = charges * unit.elementary_charge
            t0 = time.time()
            system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
            t1 = time.time()
            print(f'Preparing system complete in {(t1-t0)/60.0:.2f} minutes')
        os.chdir(current_dir)

    print('Adding restraints')
    copy_sys = add_backbone_ligand_pos_constraints(system, modeller.positions, modeller.topology.atoms(), RELAX_STIFFNESS, ligand_name, pocket_residues)
    print('Create integrator')
    integrator = LangevinIntegrator(0, 0.01, 0.0)
    simulation = Simulation(modeller.topology, copy_sys, integrator, platform=platform)
    simulation.context.setPositions(modeller.positions)

    print('Minimizing ...')
    ret = {}
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    print(f'Starting energy: {ret["einit"]:.2f}')
    t0 = time.time()
    simulation.minimizeEnergy(maxIterations=RELAX_MAX_ITERATIONS,
                              tolerance=RELAX_ENERGY_TOLERANCE)
    t1 = time.time()
    print(f'Minimizer complete in {(t1-t0)/60.0:.2f} minutes')
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    ret["min_pdb"] = _get_pdb_string(simulation.topology, state.getPositions())
    print(f'Final energy: {ret["efinal"]:.2f}')

    print(f'Wrote minimized protein to {output_prot_min}')
    prot_modeller = Modeller(modeller.topology, simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions())
    toDelete = []
    for res in prot_modeller.topology.residues():
        if res.name == ligand_name:
            toDelete.append(res)
    prot_modeller.delete(toDelete)
    with open(output_prot_min, 'w') as outfile:
        PDBFile.writeFile(prot_modeller.topology, prot_modeller.positions, file=outfile, keepIds=True)
    print('Done')


@click.command(name='minimize_protein')
@click.argument('pdb_in', type=str) # Input docking protein structure pdb to minimize (somehow openmm is slow for large protein structure, although we only allow residues in the binding pocket area to move; so it is helpful to cutoff chains that are far away from the binding pocket to speed up the process)
@click.argument('mol_in', type=str) # Ligand pose around which the docked protein structure is to be minimized; it is determined during minimization
@click.argument('output_prot_min', type=str)
@click.option('--dist_cutoff', '-cutoff', type=float, default=8.0)
@click.option('--platform', type=str, default='CPU')
def cli_minimize_protein(pdb_in, mol_in, output_prot_min, dist_cutoff, platform):
    minimize_protein(pdb_in, mol_in, output_prot_min, dist_cutoff, platform)


@click.group()
def main():
    pass

main.add_command(cli_minimize_protein)


if __name__ == '__main__':
    main()
