import pandas as pd
from abc import abstractmethod
from rdkit import Chem
import os
import logging
import getpass

log = logging.getLogger(__name__)

class Profile:

    def __init__(self, profile, activity, smiles):
        self.profile = profile
        self.activity = activity
        self.smiles = smiles

    def get_subprofile(self, aids: list, drop_null=False):
        """ returns a subset of the profile specified by aids as a Profile object

        aids: aids to keep
        drop_null: return compounds with no responses
        """
        for aid in aids:
            if aid not in self.profile.columns:
                raise Exception("{0} is not in the profile, "
                                "AIDs in profile are {1}".format(aid, self.columns.tolist()))
        sub_profile = self.profile.copy()[aids]
        if drop_null:
            profile = self.remove_nulls(sub_profile)
            return Profile(profile, self.activity[profile.index], self.smiles[profile.index])
        return Profile(sub_profile, self.activity, self.smiles)

    def remove_nulls(self, profile):
        """ remove compounds with no response in any aid """
        return profile[(profile != 0).any(1)]

class ProfileDataset:

    def __init__(self, name):
        self._id = name
        self._data_dir = os.environ.get('QSAR_DATA', None)

    def load(self):
        log.debug("Running load() on object {0}".format(self._id))
        if self._data_dir is None:
            raise Exception("QSAR_DATA environmental variable needs to be set.")


        profile_datapath = self._data_dir + str(self._id) + ".csv"
        molecule_datapath = self._data_dir + "activity_log10_scaled_activity_train.csv"
        profile = pd.read_csv(profile_datapath, index_col=0).fillna(0)

        profile.columns = list(map(int, profile.columns))
        profile.index = list(map(int, profile.index))

        molecule_data = pd.read_csv(molecule_datapath, index_col=0)
        molecule_data.index = list(map(int, molecule_data.index))
        molecule_data = molecule_data.loc[profile.index]

        return Profile(profile, molecule_data.Activity, molecule_data.SMILES)


class Datasets(object):
    """parent class for building datasets i.e. panda dataframes"""

    profile_1 = ProfileDataset("profile_1")
    profile_3 = ProfileDataset("profile_3")

    @staticmethod
    def load(ds) -> pd.DataFrame:
        return ds.load()