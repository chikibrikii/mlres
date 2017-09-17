"""
    Model classes, including deterministic rule based 
    and statistically trained machine learning models
"""

from sklearn.feature_extraction import DictVectorizer
from utils import get_targetlist, calc_accuracy
# from log_setting import logger
from data_classes import ClassifierTrainTestData
from mldb_schema import MachineLearningModels

# sklearn need to be 0.15.0, for celery use
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from models_info.params.model_parameters import MODEL_PARAMETERS
from sklearn2pmml import sklearn2pmml
from sklearn2pmml import PMMLPipeline
from config import Config
import pickle
from sklearn.preprocessing import Imputer

"""
    utility functions
"""

def models(model_parameters=MODEL_PARAMETERS):
    """
        The function that returns a dictionary of models, with the parameters set to the input parameters

        Args:
        --------
            model_parameters: a dictionary of dictionaries, sets to a specific type, which holds all models' all
            relevant parameters, default sets to MODEL_PARAMETERS

        Returns:
        --------
            A dictionary of models with parameters sets to model_parameters

        Raises:
        --------
            ValueError: If the list is not a model parameter list
    """
    if not set(model_parameters.keys()) == set(MODEL_PARAMETERS.keys()):
        raise ValueError('the input list is not model parameters list, please check the format in the format of MODEL_PARAMETERS')
    return {
        'GaussianNB': GaussianNB(**model_parameters['GaussianNB']),
        'GaussianProcess': GaussianProcessClassifier(**model_parameters['GaussianProcess']),
        'SVC_linear': SVC(**model_parameters['SVC_linear']),
        'SVC_RBF': SVC(**model_parameters['SVC_RBF']),
        'DecisionTree': DecisionTreeClassifier(**model_parameters['DecisionTree']),
        'RandomForest': RandomForestClassifier(**model_parameters['RandomForest']),
        'AdaBoost': AdaBoostClassifier(**model_parameters['AdaBoost']),
        'KNN': KNeighborsClassifier(**model_parameters['KNN']),
        'ANN_MLP': MLPClassifier(**model_parameters['ANN_MLP']),
        }


"""
    Constant definitions
"""

conf = Config()

"""
    Classes
"""

class Model():
    """Abstract class for general model.
    """
    _output_name = 'output'

    def set_output_name(self, output_name):
        self._output_name = output_name

    def compute(self, input_data):
        """ abstract method for computing the output data """
        pass


class SumOfCountModel(Model):
    """
        The simple rule based model just based on the sum of the key phrases to
        determine if the output belongs to one class or another.
    """

    _threshold = 25

    def set_threshold(self, threshold):
        self._threshold = threshold

    def compute(self, input_data):
        """
            Compute the results
        """
        label = []
        for data_point in input_data:
            label.append({self._output_name: 1 if sum([data_point[k] for k in  data_point if k != 'identifier']) >= self._threshold else 0})
        return label


class MachineLearningModel(Model):
    """
        A machine learning model is a kind of model that need to be trained before used.
    """

    def __init__(self, model_name):
        self.model_name = model_name

    def train(self, traindata):
        """Train the machine learning model using traindata """
        pass

    def test(self, testdata):
        """Test the machine learning model using testdata """
        pass


class Classifier(MachineLearningModel):
    """General classifier model """

    def __init__(self, model_name, model_parameters=MODEL_PARAMETERS):
        self.model_name = model_name
        self.classifier = models(model_parameters)[model_name]
        self.istrained = 'no'
        self.train_features = None
        self.train_features_name = None
        self.train_target = None
        self.train_target_name = None
        self.insample_acc = None
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.vectoriser = DictVectorizer(sparse=False)

    @classmethod
    def from_db(cls, dbmodel: MachineLearningModels):
        # self.model_name = dbmodel.model_name
        # self.classifier = dbmodel.model_pickle
        # self.istrained = 'from_db'
        modelfromdb = cls(dbmodel.model_name)
        modelfromdb.classifier = dbmodel.model_pickle.model_pickle
        modelfromdb.istrained = 'from_db'
        modelfromdb.train_target_name = dbmodel.train_data.target_name
        modelfromdb.train_features_name = [x.feature_name for x in dbmodel.train_data.feature_names]
        modelfromdb.imputer = dbmodel.model_pickle.imputer_pickle
        modelfromdb.vectoriser = dbmodel.model_pickle.vectoriser_pickle
        return modelfromdb

    def transform(self, data):
        return self.imputer.transform(self.vectoriser.transform(data))

    def transform_input(self, feature_pool):
        return [{k:x[k] for k in self.train_features_name} for x in feature_pool]

    def fit_transform(self, data):
        return self.imputer.fit_transform(self.vectoriser.fit_transform(data))

    def train(self, clftraindata: ClassifierTrainTestData):
        """
            use train data to train the model, also returns relevant measures

            Args:
            --------
                self: Classifier, to be trained
                clftraindata: ClassifierTrainTestData, the data used for training the model

            Returns:
            --------
                Returns nothing, but the Classifier object is trained by the training data
        """
        # self.train_features = self.vectoriser.fit_transform(clftraindata.features)
        # self.train_features = self.imputer.fit_transform(self.train_features)
        self.train_features = self.fit_transform(clftraindata.features)
        self.train_features_name = clftraindata.curr_feature_names

        # self.train_target = self.vectoriser.fit_transform(clftraindata.target)
        # self.train_target = [x[clftraindata.curr_target_name] for x in clftraindata.target]
        self.train_target = get_targetlist(clftraindata.target)
        self.train_target_name = clftraindata.curr_target_name

        self.classifier.fit(self.train_features, self.train_target)
        self.train_output = self.classifier.predict(self.train_features)

        self.istrained = 'yes'
        self.insample_acc = self.test(clftraindata)
        self.train_datapoints_id = clftraindata.get_datapoint_id()

    def test(self, testdata: ClassifierTrainTestData):
        """
            use test data to tst the model returns the accuracy w.r.t. the given dataset

            Args:
            --------
                self: Classifier, to be trained
                testdata: ClassifierTrainTestData, the data used for running a simple test the model

            Returns:
            --------
                Returns accuracy w.r.t. the given dataset
        """
        pred  = self.compute(testdata.features)
        [x.pop('probability', None) for x in pred]
        y_predicted = get_targetlist(pred, target_name=self.train_target_name)
        y_actual = get_targetlist(testdata.target, target_name=self.train_target_name)

        return calc_accuracy(y_predicted, y_actual)

    def saving_pmml(self, name=None):
        """
            Save the trained model as a PMML model

            Args:
            --------
                self: Classifier, needs to be saved as pmml.
                name: the dir + name of the .pmml file.

            Returns:
            --------
                Returns nothing but saved pmml file at given location.

            Raise:
            --------
                TypeError if the model is not trained.
        """

        if self.istrained == 'no':
            raise TypeError('the Classifier is not trained')
        pmmlpipeline = PMMLPipeline([("classifier",self.classifier)])
        file_name  = name if name else self.model_name+'.pmml'
        sklearn2pmml(pmmlpipeline, conf.MODEL_DUMP_LOCATION + file_name, with_repr = True)

    def pickle_model(self, name=None):
        """
            Save as pickle for future load

            Args:
            --------
                self: Classifier, needs to be saved as pickle.
                name: the dir + name of the pickle file.

            Returns:
            --------
                Returns nothing but saved pickle file at given location.

            Raise:
            --------
                TypeError if the model is not trained.
        """
        if self.istrained == 'no':
            raise TypeError('the Classifier is not trained')
        file_name  = name if name else self.model_name+'.pk'
        pickle.dump(self, open(conf.MODEL_DUMP_LOCATION + file_name,'wb'))

    def compute(self, input_data):
        """
            compute the result using the model giving the input data

            Args:
            --------
                self: Classifier, the model that is used for computing the output
                input_data: list of dicts of features.

            Returns:
            --------
                Returns list of the same length as inputs, of the form [{<target_name>:<target_value>}, ...]

            Raise:
            --------
                TypeError if the model is not trained.
        """

        if self.istrained == 'no':
            raise TypeError('the Classifier is not trained')
        features = self.transform(self.transform_input(input_data))
        predicted = self.classifier.predict(features)
        return [{self.train_target_name : p} for p in predicted]

    def compute_raw(self, input_data):
        """
            Get the direct predition results as the output from classifier

            Args:
            --------
                self: Classifier, the model that is used for computing the output
                input_data: list of dicts of features.

            Returns:
            --------
                Returns list of the same length as inputs, of the form [1, 2 ,3 ...]

            Raise:
            --------
                TypeError if the model is not trained.
        """

        if self.istrained == 'no':
            raise TypeError('the Classifier is not trained')
        features = self.transform(input_data)
        return self.classifier.predict(features)


class PosNegClassifier(Classifier):
    """
        Binary classifier models, the classification results are
        either positive or negative. The key difference here is that
        there is the notion for threshold, which determines how sensitive
        the classifier is.
    """

    _threshold = .5 # Threshold for triggering positive

    def set_thresh(self, threshold):
        self._threshold = threshold

    def compute(self, input_data):
        """
            In PosNegClassifier, the predicted result is the probability
            of the results being positive, and then with the threshold
            it gives back the result

            Args:
            --------
                self: PosNegClassifier, the model that is used for computing the output
                input_data: list of dicts of features.

            Returns:
            --------
                Returns list of the same length as inputs, of the form [{<target_name>:<target_value>, 'confidence':<probability_of_positive>}, ...] 

            Raise:
            --------
                TypeError if the model is not trained.
        """
        if self.istrained == 'no':
            raise TypeError('the Classifier is not trained')
        features = self.transform(input_data)
        predicted = self.classifier.predict_proba(features)

        positive_probability = predicted[:,1]
        return [{self.train_target_name : 1 if p >= self._threshold else 0, 'probability': p}\
                for p in positive_probability]

    def __repr__(self):
        return "model: %s, trained: %s, in sample accuracy: %s"\
                %(self.model_name, self.istrained, self.insample_acc)


