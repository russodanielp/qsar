from qsar.cleaner import PubChemDataSetCleaner, ActivityBalancer, \
                            StructureCleaner
from qsar.pubchem import PubChemDataSet
from qsar.datasets import Datasets as DS
from tests import skiptest



class TestPubChemDataSetCleaner:

    @skiptest
    def setup(self):
        """ PubChem AID 1224861 with
            25 actives and 109 inactives
        """
        self.ds = PubChemDataSet(1224861).load()

    def teardown(self):
        pass

    @skiptest
    def test_PubChemDataSet_cleaner(self):
        """ testing PubChemDataSet cleaner """
        pipe = PubChemDataSetCleaner(steps=[ActivityBalancer()])
        ds = pipe.run(self.ds)
        assert len((ds.Activity[ds.Activity == 1])) == 25
        assert len((ds.Activity[ds.Activity == 0])) == 25

    @skiptest
    def test_StructureChecker_cleaner(self):
        """ testing structure checker """
        ds = PubChemDataSet(1).load()
        pipe = PubChemDataSetCleaner(steps=[StructureCleaner()])
        ds = pipe.run(ds)
        assert None not in ds.rdkit.values

    def test_StructureChecker_cleaner_on_testset(self):
        """ testing to see if the code works on the test set """
        ds = DS.profile_3.load().as_ds()
        pipe = PubChemDataSetCleaner(steps=[StructureCleaner()])
        ds = pipe.run(ds)
        assert None not in ds.rdkit.values
