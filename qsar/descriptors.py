from rdkit.ML.Descriptors import MoleculeDescriptors
from .pubchem import PubChemDataSet
from rdkit.Chem import Descriptors
from rdkit import Chem
import pandas as pd


class PubChemDataSetDescriptors:
    """ a class to get descriptors from a PubChemDataSet object """

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