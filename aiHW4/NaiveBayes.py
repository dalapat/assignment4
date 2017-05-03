__author__ = 'anishdalal'
from Methods import Predictor
from collections import Counter

class NaiveBayes(Predictor):

    def __init__(self):
        self.prior = {}
        self.likelihood = {} #(label, feature vector index, feature vector attribute value)
        self.labeldict = {}
        self.denom_post = {}
        self.labelcount = {}

    def train(self, instances):
        self.prior = self.formPrior(instances)

        '''
        cat_labels_full = []
        feature_vectors_full = []
        for instance in instances:
            feature_vector = instance[0]
            cat_label = instance[1]
            feature_vectors_full.append(feature_vector)
            cat_labels_full.append(cat_label)
        fvarray = numpy.array(feature_vectors_full)
        '''
        freq_dict = {}
        freq_dict_attrs = {}
        full_cat_labels = []
        for instance in instances:
            cat_label = instance[1]
            full_cat_labels.append(cat_label)
            feature_vector = instance[0]
            for i, att_val in enumerate(feature_vector):
                key = (cat_label, i, att_val)
                key2 = (i, att_val)
                if key in freq_dict:
                    freq_dict[key] += 1
                else:
                    freq_dict[key] = 1
                if key2 in freq_dict_attrs:
                    freq_dict_attrs[key2] += 1
                else:
                    freq_dict_attrs[key2] = 1
        prob_dict = {}
        attrs_prob_dict = {}
        labelcount = self.convertSeqToCounter(full_cat_labels)
        denom = float(sum(labelcount.values()))
        self.labelcount = labelcount
        for key3 in freq_dict:
            label = key3[0]
            prob = freq_dict[key3] / float(labelcount[label])
            prob_dict[key3] = prob
        for key in freq_dict_attrs:
            attrs_prob_dict[key] = freq_dict_attrs[key] / denom
        self.denom_post = attrs_prob_dict
        self.likelihood = prob_dict

    def convertSeqToCounter(self, l):
        c = Counter()
        for e in l:
            c[e] += 1
        return c

    def formPrior(self, instances):
        c = Counter()
        cat_labels = []
        for instance in instances:
            cat_labels.append(instance[1])
        for label in cat_labels:
            c[label] += 1
        prior = {}
        self.labeldict = c.keys()
        for label_class in c.keys():
            prior[label_class] = c[label_class]/float(c.values())
        return prior

    def predict(self, instance):
        # super(NaiveBayes, self).predict(instance)
        feature_vec = instance[1]
        predicted = None
        max_prob = 0
        for label in self.labeldict:
            post = 1
            for index, attr_val in enumerate(feature_vec):
                key = (label, index, attr_val)
                key2 = (index, attr_val)
                liklihood = self.likelihood[key]
                prior = self.prior[label]
                denom = self.denom_post[key2]
                currpost = (liklihood * prior) / float(denom)
                post *= currpost
            if post > max_prob:
                max_prob = post
                predicted = label
        return predicted
