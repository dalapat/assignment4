from Methods import *
import numpy as np
from scipy.special import expit


class NeuralNetwork(Predictor):
    matrix = 0
    type = []
    input_labels = []
    input_layer_dimen = 2
    output_layer_dimen = 2
    learning_rate = 0.1
    regularization = 0.01
    relate = {}
    neural_net = 0
    examples = 0
    epoch=60000


    def __init__(self):
        self.examples = 0
        self.learning_rate = 0.005
        self.input_labels = []
        self.input_label_categories = []
        self.input_layer_dimen = 2
        self.output_layer_dimen = 2
        self.regularization = 0.05
        self.neural_net = 0
        self.type = []
        self.relate = {}
        self.matrix = 0

    def instance_iter(self, instances, outer):
        for instance in instances:
            inst_data = []
            if instance.getLabel().getLabel() not in self.type:
                self.type.append(instance.getLabel().getLabel())
                self.relate[len(self.type) - 1] = instance.getLabel().getLabel()
            self.input_labels.append(instance.getLabel().getLabel())
            for i in range(0, len(instance.getFeatureVector().getFeatureVec())):
                inst_data.append(instance.getFeatureVector().getFeatureVec()[i])
            outer.append(inst_data)
            self.input_layer_dimen = len(inst_data)
            self.output_layer_dimen = len(self.type)

    def sift_layers(self, weight_layer_1, weight_layer_2, bias_layer_1, bias_layer_2):
        output_layer_1 = self.sigmoid(self.matrix.dot(weight_layer_1) + bias_layer_1)
        scores = np.exp((self.sigmoid(np.dot(self.matrix, weight_layer_1) + bias_layer_1)).dot(weight_layer_2) + bias_layer_2)
        delt_err = scores / np.sum(scores, axis=1, keepdims=True)
        delt_err[range(self.examples), self.input_label_categories] -= 1
        dweight_layer_2 = np.dot(output_layer_1.T, delt_err)
        dweight_layer_1 = np.dot(self.matrix.T, np.dot(delt_err, weight_layer_2.T) * (1 - np.power(output_layer_1, 2)))
        return output_layer_1, dweight_layer_1, dweight_layer_2, delt_err

    def sigmoid(self,x):
        return expit(x)

    def propagate_vals(self, weight_layer_1, weight_layer_2, bias_layer_1, bias_layer_2):
        for i in range(0, 60000):
            a1, dWeight_Layer_1, d_weight_layer_2, delta = self.sift_layers(weight_layer_1, weight_layer_2, bias_layer_1, bias_layer_2)
            d_weight_layer_2 += self.regularization * weight_layer_2
            dWeight_Layer_1 += self.regularization * weight_layer_1
            weight_layer_1 = weight_layer_1 - (self.learning_rate * dWeight_Layer_1)
            bias_layer_1 = bias_layer_1 - (self.learning_rate * np.sum(np.dot(delta, weight_layer_2.T) * (1 - np.power(a1, 2)), axis=0))
            weight_layer_2 = weight_layer_2 - (self.learning_rate * d_weight_layer_2)
            bias_layer_2 = bias_layer_2 - (self.learning_rate * np.sum(delta, axis=0))
        return weight_layer_1, weight_layer_2, bias_layer_1, bias_layer_2

    def categorical_classify_input(self):
        for label in self.input_labels:
            for i in range(0, len(self.relate)):
                if label == self.relate[i]:
                    self.input_label_categories.append(i)
                    break

    def result(self, x):
        weight_layer_1 = self.neural_net['weight_layer_1']
        weight_layer_2 = self.neural_net['weight_layer_2']
        bias_layer_1 = self.neural_net['bias_layer_1']
        bias_layer_2 = self.neural_net['bias_layer_2']
        val = np.argmax((np.exp((self.sigmoid(x.dot(weight_layer_1) + bias_layer_1)).dot(weight_layer_2) + bias_layer_2))
                        / np.sum((np.exp((self.sigmoid(x.dot(weight_layer_1) + bias_layer_1)).dot(weight_layer_2) + bias_layer_2)),
                                                                                       axis=1, keepdims=True),axis=1)
        return val

    def train(self, instances):
        instance_data = []
        self.instance_iter(instances, instance_data)
        self.matrix = np.array(instance_data)
        self.examples = len(self.matrix)
        self.categorical_classify_input()

        np.random.seed(0)
        weight_layer_1 = ( 2 * np.random.randn(self.input_layer_dimen, 20) / np.sqrt(self.input_layer_dimen) ) -1
        weight_layer_2 = ( 2 * np.random.randn(20, self.output_layer_dimen) / np.sqrt(20) )-1
        bias_layer_1 = np.zeros((1, 20))
        bias_layer_2 = np.zeros((1, self.output_layer_dimen))

        weight_layer_1, weight_layer_2, bias_layer_1, bias_layer_2 = self.propagate_vals(weight_layer_1, weight_layer_2, bias_layer_1, bias_layer_2)
        
        self.neural_net = {'weight_layer_1': weight_layer_1,
                          'bias_layer_1': bias_layer_1,
                          'weight_layer_2': weight_layer_2, 'bias_layer_2': bias_layer_2}

    def predict(self, instance):
        temp = []
        for i in range(0, len(instance.getFeatureVector().getFeatureVec())):
            temp.append(instance.getFeatureVector().getFeatureVec()[i])
        return self.relate[self.result(np.array(temp))[0]]