__author__ = 'anishdalal'

class Node():

        def __init__(self, attr_label, examples):
            self.attr_label = attr_label
            self.examples = examples
            self.brances = []

        def add_branch(self, branch):
            self.brances.append(branch)
