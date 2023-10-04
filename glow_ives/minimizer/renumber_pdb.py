## Adapted from: https://github.com/rasbt/protein-science/blob/master/scripts-and-tools/renumber_pdb/renumber_pdb.py

# Sebastian Raschka 2014
# Python 3 script to atoms and residues in a PDB file.
#

class Pdb(object):
    """ Object that allows operations with protein files in PDB format. """

    def __init__(self, file_cont = [], pdb_code = ""):
        self.cont = []
        self.atom = []
        self.hetatm = []
        self.fileloc = ""
        if isinstance(file_cont, list):
            self.cont = file_cont[:]
        elif isinstance(file_cont, str):
            try:
                with open(file_cont, 'r') as pdb_file:
                    self.cont = [row.strip() for row in pdb_file.read().split('\n') if row.strip()]
            except FileNotFoundError as err:
                print(err)

        if self.cont:
             self.atom = [row for row in self.cont if row.startswith('ATOM')]
             self.hetatm = [row for row in self.cont if row.startswith('HETATM')]
             self.conect = [row for row in self.cont if row.startswith('CONECT')]

    def renumber_atoms(self, start=1):
        """ Renumbers atoms in a PDB file. """
        out = list()
        count = start
        for row in self.cont:
            if len(row) > 5:
                if row.startswith(('ATOM', 'HETATM', 'TER', 'ANISOU')):
                    num = str(count)
                    while len(num) < 5:
                        num = ' ' + num
                    row = '%s%s%s' %(row[:6], num, row[11:])
                    count += 1
            out.append(row)
        return out


    def renumber_residues(self, start=1, reset=False):
            """ Renumbers residues in a PDB file. """
            out = list()
            count = start - 1
            cur_res = ''
            for row in self.cont:
                if len(row) > 25:
                    if row.startswith(('ATOM', 'HETATM', 'TER', 'ANISOU')):
                        next_res = row[22:27].strip() # account for letters in res., e.g., '1A'

                        if next_res != cur_res:
                            count += 1
                            cur_res = next_res
                        num = str(count)
                        while len(num) < 3:
                            num = ' ' + num
                        new_row = '%s%s' %(row[:23], num)
                        while len(new_row) < 29:
                            new_row += ' '
                        xcoord = row[30:38].strip()
                        while len(xcoord) < 9:
                            xcoord = ' ' + xcoord
                        row = '%s%s%s' %(new_row, xcoord, row[38:])
                        if row.startswith('TER') and reset:
                            count = start - 1
                out.append(row)
            return out


def renumber_pdb(input_pdb, output_pdb, start, renum_atoms, renum_residues, chainreset):
    if not start:
        start = 1
    else:
        start = int(start)

    if not renum_atoms and not renum_residues:
        print('{0}\nPlease provide at least set renum_atoms or renum_residues.\n{0}'.format(50* '-'))
        quit()

    pdb1 = Pdb(input_pdb)
    if renum_atoms:
        pdb1.cont = pdb1.renumber_atoms(start=start)
    if renum_residues:
        pdb1.cont = pdb1.renumber_residues(start=start, reset=chainreset)

    with open(output_pdb, 'w') as f:
        for line in pdb1.cont:
            f.write(line + '\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Renumber residues in a pdb file',
        formatter_class=argparse.RawTextHelpFormatter
        )

    parser.add_argument('-i', '--input', help='Input PDB file')
    parser.add_argument('-o', '--output', help='Output PDB file')
    parser.add_argument('-s', '--start', help='Number of the first residue in the renumbered file (default = 1)')
    parser.add_argument('-a', '--atoms' ,action='store_true', help='Renumbers atoms')
    parser.add_argument('-r', '--residues', action='store_true', help='Renumbers residues')
    parser.add_argument('-c', '--chainreset', action='store_true', help='Resets the residue renumbering after encountering a new chain.')
    parser.add_argument('-v', '--version', action='version', version='v. 1.0')

    args = parser.parse_args()

    if not args.input:
        print('{0}\nPlease provide an input file.\n{0}'.format(50* '-'))
        parser.print_help()
        quit()

    renumber_pdb(args.input, args.output, args.start, args.atoms, args.residues, args.chainreset)
