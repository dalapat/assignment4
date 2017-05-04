__author__ = 'anishdalal'

class Node():

        def __init__(self, attr_label, examples):
            self.attr_label = attr_label
            self.examples = examples
            self.branches = []

        def getAttrLabel(self):
            return self.attr_label

        def add_branch(self, branch):
            self.branches.append(branch)
