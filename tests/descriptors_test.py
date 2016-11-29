from qsar.pubchem import PubChemDataSet
from qsar.descriptors import PubChemDataSetDescriptors
from qsar.datasets import Datasets as DS
from tests import skiptest

class TestDescriptors:

    @skiptest
    def setup(self):
        """ PubChem AID 1224861 with
            25 actives and 109 inactives
        """
        self.ds = PubChemDataSet(1).clean_load()


    def teardown(self):
        pass

    @skiptest
    def test_rdkit_descriptors(self):
        """ testing rdkit descriptors load """

        print(self.ds.rdkit.tolist())
        descriptors = PubChemDataSetDescriptors(self.ds)
        rdkit_X = descriptors.load_rdkit()
        assert 'HeavyAtomCount' in rdkit_X.columns

    def test_rdkit_descriptors_on_test_set(self):
        """ testing rdkit descriptors load on the test set """

        descriptors = PubChemDataSetDescriptors(DS.profile_3.load().as_ds())
        rdkit_X = descriptors.load_rdkit()
        assert 'HeavyAtomCount' in rdkit_X.columns

