__author__ = 'anishdalal'
import numpy as np
from Methods import Predictor
from Node import Node
import math
import sys
class DecisionTree(Predictor):


    def __init__(self):
        self.root = None
        self.examples = []
        self.distinct_labels = {}
        self.full_labels = []
        self.igr = ""

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

    def getAttributeSet(self, instances):
        fv = instances[0].getFeatureVector().getFeatureVec()
        return fv.keys()

    def getEntropy(self, subset_of_instances):
        # operates on a subset of the instances grouped by attribute value
        label_dict = self.getLabelCount(subset_of_instances)
        totalNumLabel = sum(label_dict.values())
        entropy = 0.0
        for count in label_dict.values():
            entropy += -(count / float(totalNumLabel)) * math.log((count / float(totalNumLabel)), 2)
        return entropy

    def getLabelCount(self, instances):
        label_dict = {}
        for instance in instances:
            label = instance.getLabel().getLabel()
            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1
        return label_dict

    def getInfoGain(self, attr_index, instances):
        entropy = self.getEntropy(instances)
        # group examples by attribute value
        groups = {}
        for instance in instances:
            if instance.getFeatureVector().get(attr_index) in groups:
                groups[instance.getFeatureVector().get(attr_index)].append(instance)
            else:
                groups[instance.getFeatureVector().get(attr_index)] = [instance]
        remain = 0.0
        for subset in groups.values():
            remain += (len(subset) / float(len(instances))) * self.getEntropy(subset)
        return entropy - remain

    def getImportantAttribute(self, instances):
        maxattr_index = None
        max_information_gain = -sys.maxint
        attribute_set = self.getAttributeSet(instances)
        for attribute in attribute_set:
            ig = self.getInfoGain(attribute, instances)
            if self.igr == "igr" and not ig == 0:
                ig /= self.getInstrinsicValue(attribute, self.examples)
            if ig > max_information_gain:
                max_information_gain = ig
                maxattr_index = attribute
        return maxattr_index

    def getDistinctValuesInAttribute(self, attr_index, instances):
        values = set()
        for instance in instances:
            fv = instance.getFeatureVector()
            values.add(fv.get(attr_index))
        return list(values)

    def getInstrinsicValue(self, attr_index, instances):
        values = self.getDistinctValuesInAttribute(attr_index, instances)
        iv = 0.0
        for v in values:
            p = len([e for e in instances if e.getFeatureVector().get(attr_index) == v]) / float(len(instances))
            iv += -p * math.log(p, 2)
        return iv

    '''
    def getEntropyAttribute(self, attribute_index, instances):
        asarray = self.convertInstancesToNPArray(instances)


        distinct_vals_for_attribute = {}
        vector = asarray[:, attribute_index:attribute_index+1]
        for poss_value in vector:
            # poss_value = instance.getLabel().getLabel()
            if poss_value in distinct_vals_for_attribute:
                distinct_vals_for_attribute[poss_value] += 1
            else:
                distinct_vals_for_attribute[poss_value] = 1
        numAllValues = sum(distinct_vals_for_attribute.values())
        entropy = 0.0
        for numValue in distinct_vals_for_attribute:
            entropy += -(numValue / float(numAllValues)) * math.log((numValue / float(numAllValues)), 2)

        return entropy
    '''

    def dtl(self, instances, attributes, parent_instances):
        instancesAsArray = self.convertInstancesToNPArray(instances)
        if len(instances) == 0: return self.plurality_value(parent_instances)
        if self.instancesHaveSameClassification(instancesAsArray): return instancesAsArray[0,0]
        if len(attributes) == 0: return self.plurality_value(instances)
        imp_attr_idx = self.getImportantAttribute(instances)
        node = Node(imp_attr_idx, instances)

    def train(self, instances):
        self.examples = instances

    def predict(self, instance):
        pass
