from qsar.pubchem import PubChemDataSet
from qsar.descriptors import PubChemDataSetDescriptors

class TestDescriptors:

    def setup(self):
        """ PubChem AID 1224861 with
            25 actives and 109 inactives
        """
        self.ds = PubChemDataSet(1).clean_load()


    def teardown(self):
        pass

    def test_rdkit_descriptors(self):
        """ testing rdkit descriptors load """

        print(self.ds.rdkit.tolist())
        descriptors = PubChemDataSetDescriptors(self.ds)
        rdkit_X = descriptors.load_rdkit()
        assert 'HeavyAtomCount' in rdkit_X.columns

