__author__ = 'anishdalal'
import numpy as np
from Methods import Predictor
class DecisionTree(Predictor):

    class Node():

        def __init__(self, attr_label):
            self.attr_label = attr_label
            self.examples = []
            self.brances = []

        def add_branch(self, branch):
            self.brances.append(branch)

    def __init__(self):
        self.root = None
        self.distinct_labels = {}
        self.full_labels = []

    def convertInstancesToNPArray(self, instances):
        temp = []
        for instance in instances:
            fv = instance.getFeatureVector()
            label = instance.getLabel().getLabel()
            index = 0
            row = []
            while index < len(fv):
                row.append(fv.get(index))
                index += 1
            row.insert(0, label)
            temp.append(row)
        np.unique(np.array(temp), return_counts=True)
        return np.array(temp)

    def getLabels(self, instances):
        for instance in instances:
            label = instance.getLabel().getLabel()
            if label in self.distinct_labels:
                self.distinct_labels[label] += 1
            else:
                self.distinct_labels[label] = 1
            self.full_labels.append(label)

    def getLabelFreqFromSubsetOfInstances(self, instances):
        label_freq = {}
        for instance in instances:
            cat_label = instance.getLabel().getLabel()
            if cat_label in label_freq:
                label_freq[cat_label] += 1
            else:
                label_freq[cat_label] = 1
        return label_freq

    '''
    Binning: If there are C classes, there are C bins per attribute, there are C - 1 cutoffs.
    To determine the cutoff, sort the values in the attribute column, then iterate through the values.
    Choose a given value. Choose a class. Iterate through the array until you find cutoff that contains maximum number of examples with given class.
    Once that cutoff is found, recursively descend on left side and right side to find cutoff for next class.
    Recursive stack has length C-1, so you need to pass an integer
    '''

    def getMaxOfLabelFreq(self, label_freq):
        max_freq = 0
        label = None
        for key in label_freq:
            if label_freq[key] > max_freq:
                max_freq = label_freq[key]
                label = key
        return label

    def plurality_value(self, instances):
        # select most common output values among a set of examples
        return self.getMaxOfLabelFreq(self.getLabelFreqFromSubsetOfInstances(instances))

    def instancesHaveSameClassification(self, instancesArray):
        vector = instancesArray[:, 0:1]
        u = np.unique(vector)
        if len(u) == 1: return True


    def dtl(self, instances, attributes, parent_instances):
        instancesAsArray = self.convertInstancesToNPArray(instances)
        if len(instances) == 0: return self.plurality_value(parent_instances)
        if self.instancesHaveSameClassification(instancesAsArray): return instancesAsArray[0,0]
        if len(attributes) == 0: return self.plurality_value(instances)


    def train(self, instances):
        a = np.random.rand(10)
        print np.unique(a, return_counts=True)

    def predict(self, instance):
        pass
