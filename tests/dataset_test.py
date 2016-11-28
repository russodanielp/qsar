from qsar.datasets import Datasets as DS



class TestPubChemDataSetCleaner:

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
