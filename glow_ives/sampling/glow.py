import os
import click
from pathlib import Path
from glow_ives.docking.dock import smina_dock


def glow(output_prefix, receptor_pdbqt, ligand_sdf, ref_lig_sdf):
    # Run with normal VDW repulsion
    output_file = f'{output_prefix}_normal.sdf'
    log_file = f'{output_prefix}_normal.log'
    print(f'Run GLOW with normal VDW, save to {output_file}')
    smina_dock(
        output_file, log_file, receptor_pdbqt, ligand_sdf, ref_lig_sdf,
        size=20,
        exhaustiveness=16,
        max_num_poses=1000000,
        energy_range=100000,
        min_rmsd_filter=1.5,
        mode='normal',
        )
    # Run with soften VDW repulsion
    output_file = f'{output_prefix}_soft.sdf'
    log_file = f'{output_prefix}_soft.log'
    print(f'Run GLOW with soft VDW, save to {output_file}')
    smina_dock(
        output_file, log_file, receptor_pdbqt, ligand_sdf, ref_lig_sdf,
        size=20,
        exhaustiveness=16,
        max_num_poses=1000000,
        energy_range=100000,
        min_rmsd_filter=1.5,
        mode='soft',
        )


@click.command(name='glow')
@click.argument('output_prefix', type=str)
@click.argument('receptor_pdbqt', type=str) # pdbqt file of the docking protein structure
@click.argument('ligand_sdf', type=str)     # query ligand to dock
@click.argument('ref_lig_sdf', type=str) # reference ligand pose to use as docking box center; can use the co-determined ligand pose bound in the docking protein structure
def cli_glow(output_prefix, receptor_pdbqt, ligand_sdf, ref_lig_sdf):
    glow(output_prefix, receptor_pdbqt, ligand_sdf, ref_lig_sdf)


@click.group()
def main():
    pass

main.add_command(cli_glow)


if __name__ == '__main__':
    main()
