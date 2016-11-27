from qsar.models import CreateQSARClassifcationModel, SKLearnModels
from qsar.pubchem import PubChemDataSet

from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit import Chem

from sklearn import base
import numpy as np
import pandas as pd



class TestQSARModels:

    def setup(self):
        """ sets up class with some dummy data """

        self.clfs = [clf[0] for clf in SKLearnModels.CLASSIFIERS]
        self.X = np.random.rand(100,56)
        self.y = np.random.randint(2, size=100)
        self.X_test = np.random.rand(10,56)
        self.y_test = np.random.randint(2, size=10)

    def teardown(self):
        pass

    def test_QSARClassificationModel_creation(self):
        """ test the creation of a new QSAR classifier """

        models = [CreateQSARClassifcationModel(clf)
                  for clf in self.clfs]
        assert all(base.is_classifier(mdl) for mdl in models)

    def test_QSARClassificationModel_predictions(self):
        """ tests to make sure sklearn predictions are working """

        models = [CreateQSARClassifcationModel(clf)
                  for clf in self.clfs]

        for mdl in models: mdl.fit(self.X, self.y)
        for mdl in models: print(mdl.predict(self.X_test))
        assert False

class TestFullPipe:

    def setup(self):
        """ sets up class with some dummy data """

        self.clfs = [clf[0] for clf in SKLearnModels.CLASSIFIERS]
        self.ds = PubChemDataSet(1224861).load()
        mols = [Chem.MolFromSmiles(smi) for smi in self.ds.SMILES]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])
        self.X = pd.DataFrame([list(calc.CalcDescriptors(mol)) for mol in mols],
                         columns=list(calc.GetDescriptorNames()),
                         index=self.ds.index)

    def teardown(self):
        pass

    def test_full_pipe(self):
        """ testing full pipeline """
        print(self.X)
        assert False

