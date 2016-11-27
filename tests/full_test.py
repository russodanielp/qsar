from qsar.pubchem import PubChemDataSet
from qsar.descriptors import PubChemDataSetDescriptors
from qsar.models import CreateQSARClassifcationModel, SKLearnModels
from sklearn.grid_search import GridSearchCV

class TestFull:


    def setup(self):
        self.ds = PubChemDataSet(1).clean_load()
        self.y = self.ds.Activity
        self.X = PubChemDataSetDescriptors(self.ds).load_rdkit()

    def teardown(self):
        pass

    def test_full_pipeline(self):
        models = [CreateQSARClassifcationModel(clf)
                  for clf, _ in SKLearnModels.CLASSIFIERS]

        rf = models[0]
        parameters = {
            'n_estimators': [200, 700],
            'max_features': ['auto', 'sqrt', 'log2'],
            'n_jobs':[-1]
        }

        clf = GridSearchCV(rf, parameters)
        clf.fit(self.X, self.y)
        print(clf.best_params_)
        assert False
