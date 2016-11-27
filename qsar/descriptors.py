from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.ML.Descriptors import CompoundDescriptors
from rdkit.Chem import Descriptors
from rdkit import Chem
import pandas as pd

calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])
X = pd.DataFrame([list(calc.CalcDescriptors(mol)) for mol in mols.rdkit],
                 columns=list(calc.GetDescriptorNames()),
                 index=mols.index)