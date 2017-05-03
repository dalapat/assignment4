from abc import ABCMeta, abstractmethod

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        #self.label_num = int(label)
        self.label_str = str(label)
        pass
        
    def __str__(self):
        print self.label_str
        pass

# the feature vectors will be stored in dictionaries so that they can be sparse structures
class FeatureVector:
    def __init__(self):
        self.feature_vec = {}
        pass
        
    def add(self, index, value):
        self.feature_vec[index] = value
        pass
        
    def get(self, index):
        val = self.feature_vec[index]
        return val
        

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

"""
TODO: you must implement additional data structures for
the three algorithms specified in the hw4 PDF

for example, if you want to define a data structure for the
DecisionTree algorithm, you could write

class DecisionTree(Predictor):
	# class code

Remember that if you subclass the Predictor base class, you must
include methods called train() and predict() in your subclasses
"""
