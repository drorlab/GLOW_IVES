import click
import os
import pandas as pd
import random
from tqdm import tqdm
from glob import glob
from pathlib import Path
from collections import defaultdict
import shutil
import time
import math
import tempfile
from rdkit import Chem

import parallel as par


pd.options.display.float_format = "{:,.2f}".format

from glow_ives.docking.dock import smina_dock
from glow_ives.minimizer.minimize_complex import minimize_protein
from glow_ives.utils.utils import run_cmd


@click.command(name='minimize_and_dock')
@click.argument('min_dir', type=str)
@click.argument('input_prot_pdb', type=str)
@click.argument('seed_lig_sdf', type=str)
@click.argument('query_lig_sdf', type=str)
@click.argument('ref_lig_sdf', type=str)
@click.option('--overwrite', '-ow', is_flag=True) # Overwrite existing outputs
def cli_minimize_and_dock(min_dir, input_prot_pdb, seed_lig_sdf, query_lig_sdf, ref_lig_sdf, overwrite):
    minimize_and_dock(min_dir, input_prot_pdb, seed_lig_sdf, query_lig_sdf, ref_lig_sdfoverwrite, overwrite)


def minimize_and_dock(min_dir, input_prot_pdb, seed_lig_sdf, query_lig_sdf, ref_lig_sdf, overwrite):
    try:
        min_prot_pdb = os.path.join(min_dir, 'openmm_min_prot.pdb')
        print('Minimizing pockets...')
        minimize_protein_conformer(min_dir, input_prot_pdb, seed_lig_sdf, min_prot_pdb, overwrite)
        print('Docking to minimized pockets...')
        run_docking(min_dir, min_prot_pdb, query_lig_sdf, ref_lig_sdf, overwrite)
    except Exception as e:
        print(e)
        print(f'ERROR processing {min_dir}')


def minimize_protein_conformer(min_dir, input_prot_pdb, seed_lig_sdf, min_prot_pdb, overwrite):
    if overwrite or (not os.path.exists(min_prot_pdb)):
        # Somehow smina output molecule sometimes missing hydrogens. add hydrogens back
        cmd = f'obabel {seed_lig_sdf} -O {seed_lig_sdf} -h'
        print(cmd)
        run_cmd(cmd)
        # Restrain ligand heavy atoms, minimize protein pocket side-chains (all-atoms)
        print(f'Minimizing protein {min_prot_pdb}')
        try:
            print(f'Use OPENMM minimizer')
            temp_lig = tempfile.NamedTemporaryFile(suffix='.pdb')
            minimize_protein(input_prot_pdb, seed_lig_sdf, min_prot_pdb, 8.0)
            assert os.path.exists(min_prot_pdb), f'{min_prot_pdb} does not exist'
        except Exception as e:
            print(f'ERROR minimizing protein {input_prot_pdb}. Check if the ligand bond order is correct (smina sometimes messed up with the bond order)')
            raise e


def run_docking(output_dir, min_prot_pdb, query_lig_sdf, ref_lig_sdf, overwrite):
    print(f'Docking {query_lig_sdf} to {min_prot_pdb} using {ref_lig_sdf} as center')
    assert os.path.exists(query_lig_sdf), query_lig_sdf

    # First, prepare the protein for docking. Since we started with already prepped protein
    # passed to the minimizer, we just need to convert the minimized pdb to pdbqt
    receptor_pdbqt = os.path.join(Path(min_prot_pdb).parent, f'{Path(min_prot_pdb).stem}.pdbqt')
    run_cmd(f'obabel {min_prot_pdb} -xr -O {receptor_pdbqt}')
    assert os.path.exists(receptor_pdbqt), receptor_pdbqt

    # Run docking with normal VDW repulsion
    output_prefix = os.path.join(output_dir, 'docked_poses')
    output_file = f'{output_prefix}_normal.sdf'
    log_file = f'{output_prefix}_normal.log'
    if overwrite or (not os.path.exists(output_file)):
        print(f'Run docking with normal VDW, save to {output_file}')
        smina_dock(
            output_file, log_file, receptor_pdbqt, query_lig_sdf, ref_lig_sdf,
            size=20,
            exhaustiveness=16,
            max_num_poses=300,
            min_rmsd_filter=1.5,
            mode='normal',
            )
    # Run docking with soften VDW repulsion
    output_file = f'{output_prefix}_soft.sdf'
    log_file = f'{output_prefix}_soft.log'
    if overwrite or (not os.path.exists(output_file)):
        print(f'Run docking with soft VDW, save to {output_file}')
        smina_dock(
            output_file, log_file, receptor_pdbqt, query_lig_sdf, ref_lig_sdf,
            size=20,
            exhaustiveness=16,
            max_num_poses=300,
            min_rmsd_filter=1.5,
            mode='soft',
            )


def setup(output_dir, receptor_pdb, seed_pose_sdf, num_conformations, overwrite):
    suppl = Chem.ForwardSDMolSupplier(seed_pose_sdf)
    to_process = []
    count = 0
    for i, mol in enumerate(suppl):
        if count >= num_conformations:
            break
        try:
            min_dir = os.path.join(output_dir, f'pose_{count+1:02d}')
            os.makedirs(min_dir, exist_ok=True)
            input_prot_pdb = os.path.join(min_dir, 'input_prot.pdb')
            seed_lig_sdf = os.path.join(min_dir, 'seed_lig.sdf')
            if overwrite or (not os.path.exists(input_prot_pdb)):
                shutil.copyfile(receptor_pdb, input_prot_pdb)
            if overwrite or (not os.path.exists(seed_lig_sdf)):
                writer = Chem.SDWriter(seed_lig_sdf)
                writer.write(mol)
            to_process.append((min_dir, input_prot_pdb, seed_lig_sdf))
            count += 1
        except Exception as e:
            print(f'ERROR reading seed pose {i}, skip...')
    print(f'Setup {count} seed poses in {output_dir}')
    return to_process


def ives(output_dir, receptor_pdb, receptor_pdbqt, query_lig_sdf, ref_lig_sdf, num_conformations, num_iter, overwrite, num_cpus):
    print(f'Write output to {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    # First, run docking with the soft VDW to generate the initial seed poses
    output_file = os.path.join(output_dir, f'initial_poses.sdf')
    log_file = os.path.join(output_dir, f'initial_poses.log')
    if overwrite or (not os.path.exists(output_file)):
        print(f'Generate initial seed poses, save to {output_file}')
        smina_dock(
            output_file, log_file, receptor_pdbqt, query_lig_sdf, ref_lig_sdf,
            size=20,
            exhaustiveness=16,
            max_num_poses=300,
            min_rmsd_filter=1.5,
            mode='soft',
            )
    else:
        print(f'Read initial seed poses from {output_file}')

    # Use smina score to rank the choose the seed poses. Note: can be
    # replaced with better scoring functions for more efficient sampling.
    # The smina pose output is already sorted based on their smina scores.
    seed_poses_sdf = output_file
    for iter in range(num_iter):
        iter_output_dir = os.path.join(output_dir, f'iter_{iter+1}')
        to_process = setup(iter_output_dir, receptor_pdb, seed_poses_sdf, num_conformations, overwrite)
        poses_files = []

        inputs = [(min_dir, input_prot_pdb, seed_lig_sdf, query_lig_sdf, ref_lig_sdf, overwrite) \
                  for (min_dir, input_prot_pdb, seed_lig_sdf) in to_process]
        par.submit_jobs(minimize_and_dock, inputs, num_cpus)

        for i, (min_dir, input_prot_pdb, seed_lig_sdf) in enumerate(to_process):
            normal_sdf = os.path.join(min_dir, 'docked_poses_normal.sdf')
            soft_sdf = os.path.join(min_dir, 'docked_poses_soft.sdf')
            if os.path.exists(normal_sdf):
                poses_files.append(normal_sdf)
            if os.path.exists(soft_sdf):
                poses_files.append(soft_sdf)
        # Combine the docked poses and select the next top-n seed poses for the next iteration
        output_sdf = os.path.join(output_dir, f'iter_{iter+1}_poses.sdf')
        combine_docked_poses(poses_files, output_sdf)
        seed_poses_sdf = output_sdf # Use as seed poses for the next iteration


def combine_docked_poses(poses_files, output_sdf):
    # Combine all docked poses and sort based on their smina docking scores
    # Note: can use a better scoring function to rescore and sort the poses
    pose_mols = [] # (smina_score, mol)
    for pose_sdf in tqdm(poses_files):
        suppl = Chem.ForwardSDMolSupplier(pose_sdf)
        for i, mol in enumerate(suppl):
            try:
                smina_score = mol.GetProp('minimizedAffinity')
                pose_mols.append((smina_score, mol))
            except Exception as e:
                print(f'ERROR reading pose {i} in {pose_sdf}, skip...')
        print(f'Read {i} poses from {pose_sdf}')
    # Sort based on the smina score (lower smina score is better)
    pose_mols = sorted(pose_mols, key=lambda x: x[0])
    print(f'Write {len(pose_mols)} docked poses to {output_sdf}')
    writer = Chem.SDWriter(output_sdf)
    for score, mol in tqdm(pose_mols):
        writer.write(mol)


@click.command(name='run_ives')
@click.argument('output_dir', type=str)
@click.argument('receptor_pdb', type=str)       # pdb file of the docking protein structure
@click.argument('receptor_pdbqt', type=str)     # pdbqt file of the docking protein structure
@click.argument('query_ligand_sdf', type=str)   # query ligand to dock
@click.argument('ref_lig_sdf', type=str) # reference ligand pose to use as docking box center; can use the co-determined ligand pose bound in the docking protein structure
@click.option('--num_conformations', '-num_conf', type=int, default=5) # Number of protein conformations to dock onto
@click.option('--num_iter', '-num_iter', type=int, default=1) # Number of IVES iteration to run (in practice 1 iteration is enough)
@click.option('--overwrite', '-ow', is_flag=True) # Overwrite existing outputs (including intermediates)
@click.option('--num_cpus', '-cpu', type=int, default=1) # Number of jobs to run in parallel (default: 1 (no parallelization))
def cli_ives(output_dir, receptor_pdb, receptor_pdbqt, query_ligand_sdf, ref_lig_sdf, num_conformations, num_iter, overwrite, num_cpus):
    ives(output_dir, receptor_pdb, receptor_pdbqt, query_ligand_sdf, ref_lig_sdf, num_conformations, num_iter, overwrite, num_cpus)


@click.group()
def main():
    pass

main.add_command(cli_ives)
main.add_command(cli_minimize_and_dock)


if __name__ == '__main__':
    main()
