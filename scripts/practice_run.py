from qsar.pubchem import PubChemDataSet
from qsar.descriptors import PubChemDataSetDescriptors
from qsar.datasets import Datasets as DS
from qsar.models import SKLearnModels
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

import multiprocessing
from functools import partial

from rdkit import RDLogger
import os

# suppress warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def build_models(aid, sub_directory):
    best_scores = {}
    aid_sub_directory = os.path.join(sub_directory, str(aid))

    # if its already been modeled move on
    if os.path.exists(aid_sub_directory):
        return
    else:
        os.mkdir(aid_sub_directory)
    try:
        ds = PubChemDataSet(aid).clean_load()
        ds.to_csv('{}/training.csv'.format(aid_sub_directory))
        y = ds.Activity
        X = PubChemDataSetDescriptors(ds).load_ECFP6()
        print("=======building model for aid {0}=======".format(aid))
        print("======={0} compounds: {1} active, {2} inactive=======".format(y.shape[0],
                                                                             (y == 1).sum(),
                                                                             (y == 0).sum()))

    except:
        print("error on aid {0}".format(aid))
        return

    for name, clf in SKLearnModels.CLASSIFIERS:
        pipe = Pipeline([(name, clf)])
        print("=======5-fold CV on {0}=======".format(name))
        parameters = SKLearnModels.PARAMETERS[name]
        cv_search = GridSearchCV(pipe,
                                 parameters,
                                 cv=5,
                                 scoring='accuracy',
                                 n_jobs=1,
                                 verbose=0)
        cv_search.fit(X.values, y.values)
        print("================================")
        print("The best parameters for {0} are :\n{1}".format(name,
                                                              cv_search.best_params_))
        print("The best score is {0}".format(cv_search.best_score_))
        best_scores[name] = cv_search.best_score_
        joblib.dump(cv_search.best_estimator_, '{}/{}.pkl'.format(aid_sub_directory, name))


    with open('{}/results.csv'.format(aid_sub_directory), 'w') as results_file:
        for model, score in best_scores.items():
            results_file.write(model + ',' + str(score) + '\n')


def main(aids, sub_directory="results"):
    print("Building models for {} aids".format(len(aids)))

    for aid in aids:
        build_models(aid, sub_directory=sub_directory)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Utility to build Classic ML models')
    parser.add_argument('-i', '--infile', required=True)

    args = parser.parse_args()

    with open(args.infile) as data_file:
        data = data_file.read().strip()

    aid_list = list(map(int, data.split('\n')))
    main(aid_list, 2)