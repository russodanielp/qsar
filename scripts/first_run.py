from qsar.pubchem import PubChemDataSet
from qsar.descriptors import PubChemDataSetDescriptors
from qsar.models import SKLearnModels
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

def main():
    aids = [119, 79, 83, 7, 37, 99, 129, 59, 41]

    for aid in aids:
        try:
            ds = PubChemDataSet(aid).clean_load()
            y = ds.Activity
            X = PubChemDataSetDescriptors(ds).load_rdkit()

            y = y[X.notnull().all(1)]
            X = X[X.notnull().all(1)]

            y = y[~np.isinf(X.values).any(1)]
            X = X[~np.isinf(X.values).any(1)]
            print("building model for aid {0}".format(aid))
        except:
            print("error on aid {0}".format(aid))
            continue

        for name, clf in SKLearnModels.CLASSIFIERS:

            pipe = Pipeline(list(SKLearnModels.PREPROCESS) + [(name, clf)])
            print("=======5-fold CV on {0}=======".format(name))
            parameters = SKLearnModels.PARAMETERS[name]

            cv_search = GridSearchCV(pipe,
                               parameters,
                               cv=5,
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=0)
            cv_search.fit(X.values, y.values)
            print("================================")
            print("The best parameters for {0} are :\n{1}".format(name,
                                                                  cv_search.best_params_))
            print("The best score is {0}".format(cv_search.best_score_))

if __name__ == '__main__':
    main()