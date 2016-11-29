from qsar.datasets import Datasets as DS
from tests import skiptest


class TestPubChemDataSetCleaner:

    @skiptest
    def test_profile_dataset(self):
        """ testing profile dataset loader """
        profile = DS.profile_3.load()
        assert profile.profile.shape == (7385, 1948)
        print(profile.activity.shape)
        assert profile.activity.shape == (7385,)
        assert profile.smiles.shape == (7385,)
        subset = [119, 79, 83, 7, 37, 99, 129, 59, 41]
        cluster_0 = profile.get_subprofile(subset)
        assert cluster_0.profile.shape == (7385, 9)

    def test_nulls_datasets(self):
        """ testing profile get null """
        profile = DS.profile_3.load()
        aid_119 = profile.get_subprofile([119]).get_nulls()
        assert (aid_119.profile == 0).all(1).all()