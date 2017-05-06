from collections import Counter

__author__ = 'anishdalal'
import numpy as np
from Methods import Predictor
import math
import sys

class DecisionTree(Predictor):


    def __init__(self, igr="", to_prune=False):
        self.root = {}
        self.examples = []
        self.distinct_labels = {}
        self.full_labels = []
        self.to_prune = to_prune
        self.igr = igr

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
        array = np.array(temp)
        return array

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

    def getMaxOfLabelFreq(self, label_freq):
        max_freq = 0
        label = None
        for key in label_freq:
            if label_freq[key] > max_freq:
                max_freq = label_freq[key]
                label = key
        return label

    def plurality_value(self, instances):
        return self.getMaxOfLabelFreq(self.getLabelFreqFromSubsetOfInstances(instances))

    def instancesHaveSameClassification(self, instancesArray):
        vector = instancesArray[:, 0:1]
        u = np.unique(vector)
        if len(u) == 1: return True

    def getAttributeSet(self, instances):
        fv = instances[0].getFeatureVector().getFeatureVec()
        res = fv.keys()
        sorted(res)
        return res

    def getEntropy(self, subset_of_instances):
        label_dict = self.getLabelCount(subset_of_instances)
        totalNumLabel = sum(label_dict.values())
        entropy = 0.0
        for count in label_dict.values():
            entropy += -(count / float(totalNumLabel)) * math.log((count / float(totalNumLabel)), 2)
        return entropy

    def entropy(self, label_dict):
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

    def getPartitionOfAttributeByGroup(self, attr_index):
        groups = {}
        for instance in self.examples:
            if instance.getFeatureVector().get(attr_index) in groups:
                groups[instance.getFeatureVector().get(attr_index)].append(instance)
            else:
                groups[instance.getFeatureVector().get(attr_index)] = [instance]
        return groups


    def getInfoGain(self, attr_index, instances):
        entropy = self.getEntropy(instances)
        groups = self.getPartitionOfAttributeByGroup(attr_index)
        remain = 0.0
        for subset in groups.values():
            remain += (len(subset) / float(len(instances))) * self.getEntropy(subset)
        return entropy - remain

    def getImportantAttribute(self, attribute_set, instances):
        maxattr_index = None
        max_information_gain = -sys.maxint
        ig = 0.0
        for attribute in attribute_set:
            ig = self.getInfoGain(attribute, instances)
            if self.igr == "igr" and not ig == 0:
                ig /= self.getInstrinsicValue(attribute, self.examples)
            if ig > max_information_gain:
                max_information_gain = ig
                maxattr_index = attribute
        return maxattr_index

    def getDistinctValuesInAttribute(self, attr_index):
        values = set()
        for instance in self.examples:
            fv = instance.getFeatureVector()
            values.add(fv.get(attr_index))
        return list(values)

    def getInstrinsicValue(self, attr_index, instances):
        values = self.getDistinctValuesInAttribute(attr_index)
        iv = 0.0
        for v in values:
            p = len([e for e in instances if e.getFeatureVector().get(attr_index) == v]) / float(len(instances))
            iv += -p * math.log(p, 2)
        return iv

    def dtl(self, instances, attributes, parent_instances):
        instancesAsArray = self.convertInstancesToNPArray(instances)
        if len(instances) == 0: return self.plurality_value(parent_instances)
        if self.instancesHaveSameClassification(instancesAsArray): return instancesAsArray[0,0]
        if len(attributes) == 0: return self.plurality_value(instances)
        # need to pass in current attribute set
        root = self.getImportantAttribute(attributes, instances)
        tree = {root: {}}
        values_for_attribute = self.getDistinctValuesInAttribute(root)
        groups = self.getPartitionOfAttributeByGroup(root)
        attributes.remove(root)
        for value in values_for_attribute:
            examples = groups[value]
            child = self.dtl(examples, attributes, instances)
            tree[root][value] = child
        return tree

    def pruning(self, instances, attributes, parent_instances):
        instancesAsArray = self.convertInstancesToNPArray(instances)
        if len(instances) == 0: return self.plurality_value(parent_instances)
        if self.instancesHaveSameClassification(instancesAsArray): return instancesAsArray[0,0]
        if len(attributes) == 0: return self.plurality_value(instances)
        root = self.getImportantAttribute(attributes, instances)
        tree = {root: {}}
        labels = self.getLabelFreqFromSubsetOfInstances(instances).keys()
        values_for_attribute = self.getDistinctValuesInAttribute(root)
        groups = self.getPartitionOfAttributeByGroup(root)
        attributes.remove(root)
        if self.compute_ig(instances, attributes, labels):
            for value in values_for_attribute:
                examples = groups[value]
                child = self.dtl(examples, attributes, instances)
                tree[root][value] = child
        return tree

    def compute_ig(self, instances, attributes, val):
        count = Counter()
        check = 0
        for i in val:
            count[i] += 1
        temp = self.entropy(count)
        gain = {}
        for a in attributes:
            gain[a] = self.getInfoGain(a, instances)
        for k in gain:
            if gain[k] < 0.04:
                check += 1
        if check > (len(instances) / 2):
            return False
        else:
            return True


    def train(self, instances):
        self.examples = instances
        attributes = self.getAttributeSet(instances)
        if self.to_prune:
            self.root = self.pruning(instances, attributes, instances)
        else:
            self.root = self.dtl(instances, attributes, instances)

    def print_tree(self, node):
        attr_index = node.keys()[0]
        print attr_index
        for value in node[node.keys()[0]]:
            print value, self.print_tree(node[node.keys()[0]])
        return

    def recurse(self, node, instance):
        if not isinstance(node, dict): return node
        if isinstance(node, dict) and len(node.keys()) == 1 and not isinstance(node[node.keys()[0]], dict):
            return node[node.keys()[0]]
        fv = instance.getFeatureVector()
        attr = fv.get(node.keys()[0])
        try:
            return self.recurse(node[node.keys()[0]][attr], instance)
        except KeyError:
            print node
            print node[node.keys()[0]], attr

    def findClosestBranch(self, attr, node):
        distinct_vals = self.getDistinctValuesInAttribute(attr, self.examples)
        min_dist = sys.maxint
        ret = None
        for val in distinct_vals:
            if val in node[node.keys()[0]]:
                dis = abs(val - attr)
                if dis < min_dist:
                    min_dist = dis
                    ret = val
        return ret

    def getClassificationForNodeToBePruned(self, attr_index):
        groups = self.getPartitionOfAttributeByGroup(attr_index)
        max_branch = None
        max_branch_num = 0
        for branch in groups.keys():
            if len(groups[branch]) > max_branch_num:
                max_branch_num = len(groups[branch])
                max_branch = branch
        label_freq = self.getLabelFreqFromSubsetOfInstances(groups[max_branch])
        l = self.getMaxOfLabelFreq(label_freq)
        return l

    def isLeafNode(self, node):
        key = node.keys()[0]
        for branch in node[key]:
            child = node[key][branch]
            if isinstance(child, dict):
                return False
        return True

    def predict(self, instance):
        val = self.recurse(self.root, instance)
        return val
