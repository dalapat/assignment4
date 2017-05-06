from Methods import *
import numpy as np
import math


class NeuralNetwork(Predictor):
    input_batch = 0
    labels = []
    input_labels = []
    input_layers = 2
    output_layers = 2
    output_layers = 0.1
    reg_lambda = 0.01  
    label_map = {}
    model = 0
    numInstances = 0


    def __init__(self):
        self.weight_first_layer = 0
        self.bias_first_layer = 0
        self.weight_second_layer = 0
        self.bias_second_layer = 0
        self.instancesAsNPArray = None
        self.numInstances = 0  # training set size
        self.input_layers = 2  # input layer dimensionality
        self.output_layers = 2  # output layer dimensionality
        self.output_layers = 0.002  # learning rate for gradient descent
        self.reg_lambda = 0.05  # regularization strength
        self.labels = []
        self.label_map = {}
        self.input_batch = 0
        self.input_labels = []
        self.model = 0
        self.input_label_categories = []

    def convertFeatureVectorToList(self, feature_vector):
        res = []
        for i in range(len(feature_vector)):
            res.append(feature_vector.get(i))
        return res

    def convertInstancesToNPArray(self, instances):
        mat = []
        for instance in instances:
            label = instance.getLabel().getLabel()
            fv = instance.getFeatureVector()
            fv_as_list = self.convertFeatureVectorToList(fv)
            fv_as_list.insert(0, label)
            mat.append(fv_as_list)
        return np.array(mat)

    def getLabelsFromInstances(self, instances):
        arr = self.convertInstancesToNPArray(instances)
        res_list = []
        res_dict = {}
        vector = arr[:, 0:1]
        for label in np.nditer(vector):
            label = str(label)
            res_list.append(label)
            if label in res_dict:
                res_dict[label] += 1
            else:
                res_dict[label] = 1
        return (res_list, res_dict)

    def getDataAsNPArray(self, instancesAsNPArray):
        return instancesAsNPArray[:, 1:len(instancesAsNPArray[0])]

    def train(self, instances):
        self.instancesAsNPArray = self.convertInstancesToNPArray(instances)
        self.labels, self.label_map = self.getLabelsFromInstances(instances)
        self.input_layers = len(self.instancesAsNPArray[0])
        self.output_layers = len(self.labels)
        self.input_batch = self.instancesAsNPArray[:, 1:len(self.instancesAsNPArray[0])]
        self.numInstances = len(instances)
        for label in self.input_labels:
            for i in range(0, len(self.label_map)):
                if label == self.label_map[i]:
                    self.input_label_categories.append(i)
                    break

        np.random.seed(1)
        denom = math.pow(self.input_layers, 2)
        self.weight_first_layer = (2 * np.random.randn(self.input_layers, 20) / denom) - 1
        self.weight_second_layer = (2 * np.random.randn(20, self.output_layers) / math.pow(20, 2)) - 1
        self.bias_first_layer = np.zeros((1, 20))
        self.bias_second_layer = np.zeros((1, self.output_layers))

        self.forwardProp()

    def sigmoid(self,x):
        return 1 / (1 + math.pow(math.e, -x))

    def forwardProp(self):
        for i in range(0, 60000):

            # Forward propagation
            a1 = self.sigmoid(self.input_batch.dot(self.weight_first_layer) + self.bias_first_layer)
            ret = self.sigmoid(self.input_batch.dot(self.weight_first_layer) + self.bias_first_layer)
            exp_scores = math.pow(math.e, ret.dot(self.weight_second_layer)+ self.bias_second_layer)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            delt = probs
            delt[range(self.numInstances), self.input_label_categories] -= 1
            dW2 = (a1.T).dot(delt)
            db2 = np.sum(delt, axis=0, keepdims=True)
            dW1 = np.dot(self.input_batch.T, delt.dot(self.weight_second_layer.T) * (1 - math.pow(a1, 2)))
            db1 = np.sum(delt.dot(self.weight_second_layer.T) * (1 - np.power(a1, 2)), axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.weight_second_layer
            dW1 += self.reg_lambda * self.weight_first_layer

            # Gradient descent parameter update
            self.weight_first_layer += -self.output_layers * dW1
            self.bias_first_layer += -self.output_layers * db1
            self.weight_second_layer += -self.output_layers * dW2
            self.bias_second_layer += -self.output_layers * db2


    def make_prediction(self, x):
        # W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        val = np.argmax((np.exp((self.sigmoid(x.dot(self.weight_first_layer)
                                              + self.bias_first_layer)).dot(self.weight_second_layer)
                                + self.bias_second_layer)) / np.sum((np.exp((self.sigmoid(x.dot(self.weight_first_layer)
                                + self.bias_second_layer)).dot(self.weight_second_layer)
                                + self.bias_second_layer)), axis=1, keepdims=True),axis=1)
        return val

    def predict(self, instance):
        temp = []
        for i in xrange(0, len(instance.feature_vector.feature_vec)):
            temp.append(instance.feature_vector.feature_vec[i])
            print temp
        return self.label_map[self.make_prediction(np.array(temp))[0]]