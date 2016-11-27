from qsar.cleaner import PubChemDataSetCleaner, ActivityBalancer
from qsar.pubchem import PubChemDataSet
from tests import skiptest



class TestPubChemDataSetCleaner:

    def setup(self):
        """ PubChem AID 1224861 with
            25 actives and 109 inactives
        """
        self.ds = PubChemDataSet(1224861).load()

    def teardown(self):
        pass

    def test_PubChemDataSet_cleaner(self):
        """ testing PubChemDataSet cleaner """
        pipe = PubChemDataSetCleaner(steps=[ActivityBalancer()])
        ds = pipe.run(self.ds)
        assert len((ds.Activity[ds.Activity == 1])) == 25
        assert len((ds.Activity[ds.Activity == 0])) == 25