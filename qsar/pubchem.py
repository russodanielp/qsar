"""
Module containing the class to get active and inactive compounds from PubChem

"""
from urllib import request, error
import pandas as pd
import os
import logging


log = logging.getLogger(__name__)

class PubChemDataSet:
    """ Class to interact with PubChem's PUG Rest to get datasets """

    def __init__(self, aid):

        self.aid = aid
        self.BASE = 'http://pubchem.ncbi.nlm.nih.gov/rest/pug/'


    def load(self):
        if not os.getenv('QSAR_DATA', None):
            raise Exception('No QSAR_DATA envirotnmental variable.'
                            'Please make a folder named QSAR_DATA and'
                            'set the path as an environmental variable.')

        path = os.getenv('QSAR_DATA') + 'aid_{0}.csv'.format(self.aid)
        if os.path.exists(path):
            return pd.read_csv(path, index_col=0)
        df = self.get_compounds()
        df['SMILES'] = self.get_smiles_from_cids(df.index.tolist())
        df.to_csv(path)
        return df


    def get_compounds(self) -> pd.DataFrame:
        """ returns a Pandas DataFrame """
        actives = self.get_compounds_by_cid('active')
        classes = [1] * len(actives)
        inactives = self.get_compounds_by_cid('inactive')
        classes.extend([0] * len(inactives))
        return pd.DataFrame(classes,
                            index=actives + inactives,
                            columns=['Activity'])

    def get_smiles_from_cids(self, cids: list) -> list:
        """ get smiles from a given list of cids """

        cids = list(map(str, cids))
        chunks = [cids[i:i + 100] for i in range(0, len(cids), 100)]
        chunks_of_smiles = []
        for chunk in chunks:
            url  = '{0}compound/cid/{1}/property/CanonicalSMILES/txt'.format(self.BASE,
                                                                              ','.join(chunk))
            log.debug(url)
            try:
                txt = request.urlopen(url)
                smiles = [smile.decode('utf-8').strip() for smile in txt]
            except error.HTTPError as err:
                log.error(err)
                smiles = []
            except error.URLError as err:
                log.error(err)
                smiles = []
            except TimeoutError as err:
                log.error(err)
                smiles = []
            chunks_of_smiles.append(smiles)

        import itertools

        return list(itertools.chain.from_iterable(chunks_of_smiles))

    def get_compounds_by_cid(self, classification: str):
        """ get compounds from a given aid by their activity classification """

        url  = '{0}assay/aid/{1}/cids/TXT?cids_type={2}'.format(self.BASE, self.aid, classification)
        try:
            txt = request.urlopen(url)
            cids = [cid.decode('utf-8').strip() for cid in txt]
            cids = list(map(int, cids))
        except error.HTTPError as err:
            cids = []
        except error.URLError as err:
            cids = []
        except TimeoutError:
            cids = []
        if 0 in cids:
            cids.remove(0)
        return cids

