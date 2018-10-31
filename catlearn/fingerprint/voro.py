"""Voronoi fingerprint based on Magpie."""
from __future__ import absolute_import
from __future__ import division

import pandas as pd
import os


class VoronoiFingerprintGenerator(object):
    """Function to build voronoi fingerprint in pandas.DataFrame.

    Based on a list of ase.atoms object.
    """

    def __init__(self, atoms, delete_temp=True):
        """Initialize voronoi generator.

        Parameters
        __________
        atoms: list
            list of structures in ase.atoms.
        """
        self.atoms = atoms
        self.temp_path = 'voro_temp'
        from catlearn import __path__
        if os.path.exists(__path__[0] + '/api/magpie'):
            self.magpie_path = __path__[0] + '/api/magpie'
        else:
            raise EnvironmentError('Magpie path not exist!')
        self.cif_path = self.temp_path + '/cif/'
        self.voro_input = '''data = new data.materials.CrystalStructureDataset
data attributes properties directory %s/lookup-data
data attributes properties add set general
data import ./%s
data target %s
data attributes generate
save data %s/voro_FP csv
exit''' % (self.magpie_path, self.cif_path, 'id', self.temp_path)
        self.magpie = 'java -jar %s/Magpie.jar' % self.magpie_path
        self.delete_temp = delete_temp

    def write_voro_input(self):
        """Write Voronoi input for Magpie."""
        if not os.path.exists(self.temp_path):
            os.mkdir(self.temp_path)
        else:
            import shutil
            shutil.rmtree(self.temp_path)
            os.mkdir(self.temp_path)
        os.mkdir(self.cif_path)
        i = 0
        filename = []
        pro_dict = {}
        fml = []
        id_list = []
        for at in self.atoms:
            at_name = str(i) + '-' + at.get_chemical_formula()
            at.write(self.cif_path + at_name,
                     format='vasp', direct=True, vasp5=True)
            filename.append(at_name)
            fml.append(at.get_chemical_formula())
            id_list.append(i)
            i += 1
        pro_dict['filename'] = filename
        pro_dict['formula'] = fml
        pro_dict['id'] = id_list
        temp_pd = pd.DataFrame.from_dict(pro_dict)
        temp_pd = temp_pd[['filename', 'id', 'formula']]
        self.input_pd = temp_pd
        temp_pd.to_csv(self.cif_path + 'properties.txt', sep=' ', index=False)
        f = open(self.temp_path + '/voro_FP.in', 'w')
        f.writelines(self.voro_input)
        f.close()

    def run_voro(self):
        """Call Magpie to generate Voronoi FP and write to voro_FP.csv."""
        os.system("%s %s/voro_FP.in |tee %s/voro_FP.log" % (
            self.magpie, self.temp_path, self.temp_path))

    def generate(self):
        """Generate Voronoi fingerprint and return all the fingerprint.

        Returns
        -------
        FP : pandas.Frame
        """
        print('Generate Voronoi fingerprint of %d structures' % len(
            self.atoms))
        self.write_voro_input()
        self.run_voro()
        try:
            FP = pd.read_csv(self.temp_path + '/voro_FP.csv')
        except:
            raise EnvironmentError(
                'Please install Java! https://java.com/en/download/')
        if self.delete_temp:
            import shutil
            shutil.rmtree(self.temp_path)
        return pd.merge(self.input_pd, FP, left_on='id', right_on='id').drop(['filename','formula','id'], axis=1)
