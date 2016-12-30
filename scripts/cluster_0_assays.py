from qsar.pubchem import PubChemDataSet
from qsar.descriptors import PubChemDataSetDescriptors
from qsar.datasets import Datasets as DS
from qsar.models import SKLearnModels
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd

import numpy as np

from rdkit import RDLogger

# suppress warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def main():
    aids = [119, 79, 83, 7, 37, 99, 129, 59, 41]
    profile = DS.profile_3.load()
    best_models = {}
    predictions = []
    for aid in aids:
        try:
            ds = PubChemDataSet(aid).clean_load()
            y = ds.Activity
            X = PubChemDataSetDescriptors(ds).load_rdkit()

            # TODO: put this into a cleaner step
            # remove null values
            y = y[X.notnull().all(1)]
            X = X[X.notnull().all(1)]

            # TODO: put this into a cleaner step
            # remove null values
            y = y[~np.isinf(X.values).any(1)]
            X = X[~np.isinf(X.values).any(1)]
            print("=======building model for aid {0}=======".format(aid))
            print("======={0} compounds: {1} active, {2} inactive=======".format(y.shape[0],
                                                                                 (y == 1).sum(),
                                                                                 (y == 0).sum()))
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
            best_models[aid] = cv_search.best_estimator_

        ds_test = profile.get_subprofile([aid]).get_nulls().as_ds()
        X_test = PubChemDataSetDescriptors(ds_test).load_rdkit()

        # save null or inf values
        dropped_cmps = X_test[~(X_test.notnull().all(1)) | (np.isinf(X_test.values).any(1))]

        # remove null and inf values
        X_test = X_test[X_test.notnull().all(1)]
        X_test = X_test[~np.isinf(X_test.values).any(1)]
        print("Making predictions on {0} compounds".format(X_test.shape[0]))
        preds = pd.DataFrame(cv_search.predict(X_test), index=X_test.index, columns=[aid])
        predictions.append(preds)
    print(pd.concat(predictions, axis=1))
    import os
    filename = os.getenv('QSAR_DATA') + 'missing_data_predictions.csv'
    pd.concat(predictions, axis=1).to_csv(filename)

if __name__ == '__main__':
    main()