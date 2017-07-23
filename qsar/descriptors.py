from rdkit.ML.Descriptors import MoleculeDescriptors
from .pubchem import PubChemDataSet
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import Chem
import pandas as pd




class PubChemDataSetDescriptors:
    """ a class to get descriptors from a dataset dataframe

     ds should be a pandas DataFrame with a column labeled smiles
     """

    def __init__(self, ds):
        self.ds = ds


    def load_rdkit(self):
        ds = self.ds.copy()
        ds['rdkit'] = [Chem.MolFromSmiles(smi) if Chem.MolFromSmiles(smi) else None
                       for smi in ds.SMILES]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])
        X = pd.DataFrame([list(calc.CalcDescriptors(mol)) for mol in ds.rdkit],
                         columns=list(calc.GetDescriptorNames()),
                         index=ds.index)
        return X

    def load_ECFP6(self):
        ds = self.ds.copy()
        ds['rdkit'] = [Chem.MolFromSmiles(smi) if Chem.MolFromSmiles(smi) else None
                       for smi in ds.SMILES]

        data = []
        for mol in ds.rdkit:
            data.append([int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024)])
        return pd.DataFrame(data, columns=list(range(1024)), index=ds.index)