import os
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import tempfile
from pathlib import Path
from glob import glob
import timeit

from glow_ives.utils.utils import run_cmd

SOFT_WEIGHTS = '''
-0.035579    gauss(o=0,_w=0.5,_c=8)
-0.005156    gauss(o=3,_w=2,_c=8)
0.2          repulsion(o=0,_c=8)
-0.035069    hydrophobic(g=0.5,_b=1.5,_c=8)
-0.587439    non_dir_h_bond(g=-0.7,_b=0,_c=8)
1.923        num_tors_div
'''


def compute_ligand_heavy_atom_center(lig_sdf):
    from rdkit import Chem
    import re
    if re.search(r'.pdb$', lig_sdf):
        mol = Chem.MolFromPDBFile(lig_sdf, removeHs=True)
    elif re.search(r'.mol2$', lig_sdf):
        mol = Chem.MolFromMol2File(lig_sdf, removeHs=True)
    elif re.search(r'.sdf$', lig_sdf):
        mol = Chem.MolFromMolFile(lig_sdf, removeHs=True)
    elif re.search(r'.mae$', lig_sdf):
        suppl = Chem.MaeMolSupplier(lig_sdf, removeHs=True)
        mol = [mol for mol in suppl][0]
    else:
        raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")

    heavy_atom_coords = mol.GetConformer().GetPositions()
    center = heavy_atom_coords.mean(axis=0)
    return center


@click.command(name='smina_dock')
@click.argument('output_sdf', type=str)
@click.argument('log_file', type=str)
@click.argument('receptor_pdbqt', type=str)
@click.argument('ligand_sdf', type=str)
@click.argument('ref_lig_sdf', type=str)
@click.option('--size', '-size', type=float, default=20)
@click.option('--exhaustiveness', '-ex', type=int, default=16)
@click.option('--energy_range', '-energy', type=float, default=3)
@click.option('--max_num_poses', '-mpose', type=int, default=300)
@click.option('--min_rmsd_filter', '-min_rmsd', type=float, default=1.5)
@click.option('--mode', type=click.Choice(['normal', 'soft']), default='normal')
def cli_smina_dock(output_sdf, log_file, receptor_pdbqt, ligand_sdf, ref_lig_sdf, size, exhaustiveness,
                   energy_range, max_num_poses, min_rmsd_filter, mode):
    smina_dock(output_sdf, log_file, receptor_pdbqt, ligand_sdf, ref_lig_sdf, size, exhaustiveness,
               energy_range, max_num_poses, min_rmsd_filter, mode)


def smina_dock(
        output_sdf, log_file, receptor_pdbqt, ligand_sdf, ref_lig_sdf,
        size=20, exhaustiveness=16, energy_range=3, max_num_poses=100, min_rmsd_filter=1.5,
        mode='normal'):

    # Compute center for docking box
    box_center = compute_ligand_heavy_atom_center(ref_lig_sdf)

    # Run docking
    cmd = f'smina --cpu 1 -r {receptor_pdbqt} -l {ligand_sdf} ' \
          f'--center_x {box_center[0]:.4f} --center_y {box_center[1]:.4f} --center_z {box_center[2]:.4f} ' \
          f'--size_x {size} --size_y {size} --size_z {size} --min_rmsd_filter {min_rmsd_filter} ' \
          f'--exhaustiveness {exhaustiveness} --energy_range {energy_range} -o {output_sdf} --num_modes {max_num_poses} --log {log_file}'

    if mode == 'soft':
        soft_weights = tempfile.NamedTemporaryFile(suffix='.txt')
        soft_weights_filename = soft_weights.name
        with open(soft_weights_filename, 'w') as f:
            print(SOFT_WEIGHTS, file=f)
        cmd += f' --custom_scoring {soft_weights_filename}'
    else:
        assert mode=='normal', f'Unrecognized mode: {mode}'
    print(cmd)
    tic = timeit.default_timer()
    out = os.popen(cmd).read()
    assert os.path.exists(output_sdf), f'{output_sdf} does not exist'
    # Somehow the output molecule sometimes missing hydrogens. add hydrogens back
    cmd = f'obabel {output_sdf} -O {output_sdf} -h'
    print(cmd)
    run_cmd(cmd, f'Failed to run smina dock {output_sdf:}')
    toc = timeit.default_timer()
    elapsed = toc - tic # in seconds
    print(f'Run smina docking in {elapsed/60.0:.3f} minutes')


@click.group()
def main():
    pass

main.add_command(cli_smina_dock)


if __name__ == '__main__':
    main()
