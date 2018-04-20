"""Prototype fingerprint based on Magpie."""
from __future__ import absolute_import
from __future__ import division

import pandas as pd
import os


class PrototypeSites(object):
    """Prototype site objective for generating prototype input."""

    def __init__(self, site_dict=None):
        if site_dict is None:
            site_dict = {}
            # default is perovskite
            site_dict['A'] = [1]
            site_dict['B'] = [1]
            site_dict['C'] = [3, '-omit']
        self.site_dict = site_dict
        self.site_list = site_dict.keys()
        self.site_list.sort()
        temp_str = []
        for si in self.site_list:
            temp_str.append(' '.join([str(x) for x in self.site_dict[si]]))
        self.site_str = '\n'.join(temp_str)


class PrototypeFingerprintGenerator(object):
    """Function to build prototype fingerprint in pandas.DataFrame.

    Based on a list of ase.atoms object.
    """

    def __init__(self, atoms, sites, system_name='', target='id',
                 delete_temp=True, properties=[]):
        """Initialize voronoi generator.

        Parameters
        __________
        atoms: list
            list of structures in ase.atoms.
        sites: PrototypeSites
            PrototypeSites including all site informations.
        """
        self.atoms = atoms
        self.sites = sites
        self.system_name = system_name
        self.temp_path = 'proto_temp' + system_name
        from catlearn import __path__
        if os.path.exists(__path__[0] + '/api/magpie'):
            self.magpie_path = __path__[0] + '/api/magpie'
        else:
            raise EnvironmentError('Magpie path not exist!')
        self.target = target
        self.properties = properties
        self.txt_path = self.temp_path + '/prototypes.txt'
        self.proto_input = '''data = new data.materials.PrototypeDataset %s/site-info.txt
data attributes properties directory %s/lookup-data
data attributes properties add set general
data import ./%s
data target %s
data attributes generate
save data ./%s/proto_FP csv
exit''' % (self.temp_path, self.magpie_path, self.txt_path, self.target,
           self.temp_path)
        self.magpie = 'java -jar %s/Magpie.jar' % self.magpie_path
        self.delete_temp = delete_temp

    def write_proto_input(self):
        """Write Prototype input for Magpie."""
        if not os.path.exists(self.temp_path):
            os.mkdir(self.temp_path)
        else:
            import shutil
            shutil.rmtree(self.temp_path)
            os.mkdir(self.temp_path)
        pro_dict = {}
        for pro in self.properties:
            pro_dict[pro] = []
        # check whether all sites are given in atoms
        for si in self.sites.site_list:
            if si not in self.properties:
                raise ValueError('No information for %s' % si)

        fml = []
        id_list = []
        i = 0
        for at in self.atoms:
            at_name = ''
            for si in self.sites.site_list:
                at_name = at_name + \
                    getattr(at, si) + str(self.sites.site_dict[si][0])
            at_name = at_name.replace('1', '')
            fml.append(at_name)
            for pro in self.properties:
                if hasattr(at, pro):
                    pro_dict[pro].append(getattr(at, pro))
                else:
                    pro_dict[pro].append(None)
            id_list.append(i)
            i += 1
        pro_dict['formula'] = fml
        pro_dict['id'] = id_list
        temp_pd = pd.DataFrame.from_dict(pro_dict)
        temp_pd = temp_pd[['formula', 'id'] + self.properties]
        self.input_pd = temp_pd
        temp_pd.to_csv(self.txt_path, sep=' ', index=False)
        f = open(self.temp_path + '/prototype_FP.in', 'w')
        f.writelines(self.proto_input)
        f.close()
        f = open(self.temp_path + '/site-info.txt', 'w')
        f.write(self.sites.site_str)
        f.close()

    def run_proto(self):
        """Call Magpie to generate Prototype FP and write to proto_FP.csv."""
        os.system("%s %s/prototype_FP.in |tee %s/prototype_FP.log" % (
            self.magpie, self.temp_path, self.temp_path))

    def update_str(self):
        self.proto_input = '''data = new data.materials.PrototypeDataset %s/site-info.txt
        data attributes properties directory %s/lookup-data
        data attributes properties add set general
        data import ./%s
        data target %s
        data attributes generate
        save data ./%s/proto_FP csv
        exit''' % (self.temp_path, self.magpie_path, self.txt_path,
                   self.target, self.temp_path)

    def generate(self):
        """Generate Prototype fingerprint and return all the fingerprint.

        Returns
        -------
        FP : pandas.Frame
        """
        print('Generate Prototype fingerprint of %d structures' %
              len(self.atoms))
        self.update_str()
        self.write_proto_input()
        self.run_proto()
        try:
            FP = pd.read_csv(self.temp_path + '/proto_FP.csv')
        except:
            raise EnvironmentError(
                'Pelase install Java! https://java.com/en/download/')
        if self.delete_temp:
            import shutil
            shutil.rmtree(self.temp_path)
        return FP

    def generate_all(self):
        self.target = 'id'
        temp_FP = self.generate()
        return pd.merge(self.input_pd, temp_FP, left_on='id', right_on='id')
