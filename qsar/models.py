from sklearn.base import ClassifierMixin, BaseEstimator

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import base

import logging

log = logging.getLogger(__name__)

class SKLearnModels:

    CLASSIFIERS = (
        (RandomForestClassifier, 'Random Forest'),
        (SVC, 'Support Vector Classification'),
        (GaussianNB, 'Naive-Bayes')
    )

    DESCRIPTORS = (
        'Rdkit'
    )

def CreateQSARClassifcationModel(classifier):
    """ function that generates a QSAR classifcation model from a sklearn model """

    class QSARClassificationModel(classifier):
        """ Class that inherits properties of sklearn classifiers """

        def __init__(self):
            params = classifier().get_params()
            classifier.__init__(self, **params)

        def __str__(self):
            return "<QSAR Model>"

        def __repr__(self):
            return self.__str__()

    # returns an instance of an QSARClassificationModel
    # not sure if it is more pythonic to find a way
    # to return an uninstantiated object
    return QSARClassificationModel


# #
# # class ConsensusModel:
# #     """ container class for all QSARClassifcationModels """
# #
# #     rf_rdkit = QSARRFClassification('rdkit')
# #     svm_rdkit = QSARSVClassification('rdkit')
# #     nb_rdkit = QSARNBClassification('rdkit')
# #
# #
# #     def __str__(self):
# #         s = 'Consensus model consisting of {0} models\n'.format(len(self._models))
# #         for model in self._models:
# #             s += str(model)+'\n'
# #         return s
# #
# #     @staticmethod
# #     def fit(model, X, y):
# #         return model.fit(X, y)