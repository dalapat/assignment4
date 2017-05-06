import numpy as np
from Methods import Predictor


class NaiveBayes(Predictor):

    def __init__(self):
        self.likelihoods = {}
        self.class_priors = {}

    def train(self, instances):
        classes = {}
        # Data pre-processing
        for instance in instances:
            label = instance.getLabel().getLabel()
            if label not in self.class_priors:
                self.class_priors[label] = 0
                self.likelihoods[label] = {}
                classes[label] = []
            classes[label].append(instance)
            self.class_priors[label] += 1
        features = instances[0].getFeatureVector().getFeatureVec().keys()
        # Training model
        for label in self.class_priors:
            self.class_priors[label] = self.class_priors[label] / float(len(instances))
            for feature in features:
                mean = self._mean(classes[label], feature)
                variance = self._variance(classes[label], feature, mean)
                self.likelihoods[label][feature] = mean, variance

    def _mean(self, instances, feature):
        s = 0
        for instance in instances:
            s += instance.getFeatureVector().getFeatureVec()[feature]
        return (1/float(len(instances))) * s

    def _variance(self, instances, feature, mean):
        sl = 0
        for instance in instances:
            sl += (instance.getFeatureVector().getFeatureVec()[feature] - mean) ** 2
        return (1/float(len(instances) - 1)) * sl

    def _gaussian(self, value, mv):
        mean, variance = mv
        gaussian = (1/np.sqrt(2*np.pi*variance)) * np.exp((-1/(2*variance))*(value - mean)**2)
        return gaussian

    def predict(self, instance):
        posteriors = {}
        features = instance.getFeatureVector().getFeatureVec().keys()
        for label in self.class_priors:
            posteriors[label] = self.class_priors[label]
            for feature in features:
                posteriors[label] *= self._gaussian(instance.getFeatureVector().getFeatureVec()[feature],
                                                    self.likelihoods[label][feature])
        return max(posteriors, key=posteriors.get)
