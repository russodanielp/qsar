from qsar.pubchem import PubChemDataSet
from qsar.descriptors import PubChemDataSetDescriptors
from qsar.models import CreateQSARClassifcationModel, SKLearnModels
from sklearn.grid_search import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import numpy as np

class TestFull:


    def setup(self):
        self.ds = PubChemDataSet(1).clean_load()
        self.y = self.ds.Activity
        self.X = PubChemDataSetDescriptors(self.ds).load_rdkit()
        #print(self.X.shape, self.y.shape)

        self.y = self.y[self.X.notnull().all(1)]
        self.X =  self.X[self.X.notnull().all(1)]
        #print(self.X.shape, self.y.shape)

        self.y = self.y[~np.isinf(self.X.values).any(1)]
        self.X = self.X[~np.isinf(self.X.values).any(1)]
        #print(self.X.shape, self.y.shape)



    def teardown(self):
        pass

    def test_full_pipeline(self):

        pipe = Pipeline(list(SKLearnModels.PREPROCESS))


        classifier = SKLearnModels.CLASSIFIERS[0]
        pipe.steps.append(classifier)
        parameters = SKLearnModels.PARAMETERS[classifier[0]]

        cv_search = GridSearchCV(pipe,
                           parameters,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=0)
        cv_search.fit(self.X.values, self.y.values)
        print(cv_search.best_params_, cv_search.best_score_)
        assert False
