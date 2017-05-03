__author__ = 'anishdalal'

from Methods import Predictor
import scipy.stats

class NaiveBayes(Predictor):

    def __init__(self):
        self.distinct_classes = {}
        self.all_classes = []
        self.prior = {}
        self.mean_for_attribute = {} #(class, attribute_index)
        self.var_for_attribute = {} #(class, attribute_index)
        self.num_instances = 0

    def getDistinctClasses(self, instances):

        for instance in instances:
            cat_label_obj = instance.getLabel()
            cat_label_s = cat_label_obj.getLabel()
            if cat_label_s in self.distinct_classes:
                self.distinct_classes[cat_label_s] += 1
            else:
                self.distinct_classes[cat_label_s] = 1
            self.all_classes.append(cat_label_s)
        self.num_instances = len(self.all_classes)

    def convertFeatureVecToList(self, feature_vec):
        res = []
        for key in feature_vec:
            res.append(feature_vec[key])
        return res

    def convertFeatureVectorsToMatrix(self, instances):
        res = []
        for instance in instances:
            feature_vec = instance.getFeatureVector().getFeatureVec()
            res.append(self.convertFeatureVecToList(feature_vec))
        return res

    def setMeanVarForAttribute(self, instances):
        mat = self.convertFeatureVectorsToMatrix(instances)
        c_freq_dict = {} #col -> {C1:sum1, C2:sum2, ... CN:sumN}
        partition_class_by_col = {} # col -> {C1: [val1, val2], C2: [val1, val2, val...]}
        for row in range(len(mat)):
            for col in range(len(mat[0])):
                if col in c_freq_dict:
                    c_freq_dict[col][self.all_classes[row]] += mat[row][col]
                else:
                    val = {}
                    for c in self.distinct_classes:
                        val[c] = 0
                    val[self.all_classes[row]] = mat[row][col]
                    c_freq_dict[col] = val
                if col in partition_class_by_col:
                    partition_class_by_col[col][self.all_classes[row]].append(mat[row][col])
                else:
                    val = {}
                    for c in self.distinct_classes:
                        val[c] = []
                    val[self.all_classes[row]].append(mat[row][col])
                    partition_class_by_col[col] = val

        for coln in c_freq_dict:
            val = c_freq_dict[coln]
            for c in val:
                self.mean_for_attribute[(c, coln)] = val[c] / float(self.distinct_classes[c])

        for colnm in partition_class_by_col:
            val = partition_class_by_col[colnm]
            for c in val:
                mean = self.mean_for_attribute[(c, colnm)]
                vals = val[c]
                variance = sum([((mean - value)**2) for value in vals]) / float(len(vals))
                self.var_for_attribute[(c, colnm)] = variance

    def calculatePriors(self):
        for label in self.distinct_classes:
            self.prior[label] = self.distinct_classes[label] / float(self.num_instances)

    def calculateLiklihoodSingle(self, value, label, mean, variance):
        return scipy.stats.norm(mean, variance).pdf(value)

    def train(self, instances):
        self.getDistinctClasses(instances)
        self.setMeanVarForAttribute(instances)
        self.calculatePriors()

    def predict(self, instance):
        fv = instance.getFeatureVector().getFeatureVec()
        label = instance.getLabel().getLabel()
        listfv = self.convertFeatureVecToList(fv)
        max_post = 0
        predicted = None
        for label in self.distinct_classes:
            post = 1
            for i, e in enumerate(listfv):
                mean = self.mean_for_attribute[(label, i)]
                variance = self.var_for_attribute[(label, i)]
                liklihood = self.calculateLiklihoodSingle(e, label, mean, variance)
                post *= liklihood * self.prior[label]
                if post > max_post:
                    max_post = post
                    predicted = label

        return predicted