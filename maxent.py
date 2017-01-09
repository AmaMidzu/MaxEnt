# -*- mode: Python; coding: utf-8 -*-

import numpy as np
import scipy.misc
from classifier import Classifier

class MaxEnt(Classifier):

    def __init__(self, model=None):
        super(Classifier, self).__init__()
        self.parameters = None # weights matrix of feature x class size
        # self.log_parameters = None # matrix for negative log-likelihood
        self.feature_vector = []
        self.classes = []

    def get_model(self): return self.parameters
    def set_model(self, model): self.parameters = model
    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        index = int(len(instances)*0.8)
        dev_instances = instances[index:]
        instances = instances[:index]
        self.train_sgd(instances, dev_instances, 0.0001, 30)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient"""
        start = 0
        count = 0
        prev_acc = 0
        acc = 0
        # prev_nll = 0
        # nll = 0
        while prev_acc >= acc: # check the convergence by checking the accuracy on the dev-set
        # while prev_nll <= nll: # can also check the convergence indirectly by checking if the negative log-likelihood stopped decreasing
            # compute gradient and parameters for each batch
            batch = train_instances[start:batch_size]
            # extract features and classes
            for instance in batch:
                feats = []
                if len(instance.features()) > 10:
                    feats = instance.features()[:10]
                else: feats = instance.features()
                for f in feats:
                    if f not in self.feature_vector: self.feature_vector.append(f)
                if instance.label not in self.classes: self.classes.append(instance.label)

            # weights matrix of feature x class size. Init to 0s.
            self.parameters = np.zeros(shape=(len(self.feature_vector), len(self.classes)))

            self.log_parameters = np.zeros(shape=(len(self.feature_vector), len(self.classes)))

            # get empirical counts for feature-class
            emp = np.zeros(shape=(len(self.feature_vector), len(self.classes)))
            for instance in batch:
                feats = []
                if len(instance.features()) > 10:
                    feats = instance.features()[:10]
                else: feats = instance.features()
                for f in feats:
                    if f in self.feature_vector:
                        emp[self.feature_vector.index(f)][self.classes.index(instance.label)] += 1

            # compute posteriors and store in a matrix, too
            posterior = np.zeros(shape=(len(self.feature_vector), len(self.classes)))
            for instance in batch:
                feats = []
                if len(instance.features()) > 10:
                    feats = instance.features()[:10]
                else: feats = instance.features()
                for f in feats:
                    denom = 0
                    for k in range(0,len(self.classes)):
                        # denom += np.exp(self.parameters[self.feature_vector.index(f)][k])
                        denom = scipy.misc.logsumexp(self.parameters[self.feature_vector.index(f)][k])
                    # posterior[self.feature_vector.index(f)][self.classes.index(instance.label)] = (np.exp(self.parameters[self.feature_vector.index(f)][self.classes.index(instance.label)]))/denom
                    posterior[self.feature_vector.index(f)][self.classes.index(instance.label)] = np.exp(self.parameters[self.feature_vector.index(f)][self.classes.index(instance.label)]) - denom

            # compute expected counts
            exp = np.zeros(shape=(len(self.feature_vector), len(self.classes)))
            for i in range(0,len(self.feature_vector)):
                for j in range(0,len(self.classes)):
                    for k in range(0,len(self.classes)):
                        exp[i][j] += emp[i][k]*posterior[i][k]

            # compute gradient vector for each class => matrix representation
            gradient = np.zeros(shape=(len(self.feature_vector), len(self.classes)))
            for i in range(0,len(self.feature_vector)):
                for j in range(0,len(self.classes)):
                    gradient[i][j] = emp[i][j] - exp[i][j]

            # update weights
            # prev_nll = nll
            for i in range(0,len(self.feature_vector)):
                for j in range(0,len(self.classes)):
                    self.parameters[i][j] += learning_rate*gradient[i][j]
                    # self.log_parameters[i][j] = scipy.log(self.parameters[i][j])
            # nll = (-1)*np.ma.masked_invalid(self.log_parameters).sum() # masked array to avoid numerical overflow

            # update indeces for next batch
            start = batch_size
            batch_size += batch_size

            # compute accuracy on dev-set
            for instance in dev_instances:
                res = 0
                c = 0
                feats = instance.features()
                feats = []
                if len(instance.features()) > 10:
                    feats = instance.features()[:10]
                else: feats = instance.features()
                # compute sigmoid for each class
                for k in range(0,len(self.classes)):
                    z = 0
                    for f in feats:
                        if f in self.feature_vector:
                            z += self.parameters[self.feature_vector.index(f)][k]
                    sigmoid = 1.0 / (1.0 + np.exp(-1.0*z))
                    if sigmoid >= res:
                        c = k
                        res = sigmoid
                # return the highest (== the most probable)
                if self.classes[c] == instance.label:
                    count +=1
            prev_acc = acc
            acc = float(count)/len(dev_instances)
            # print acc

    def classify(self, instance):
        res = 0
        c = 0
        feats = instance.features()
        # compute sigmoid for each class
        for k in range(0,len(self.classes)):
            z = 0
            for f in feats:
                if f in self.feature_vector:
                    z += self.parameters[self.feature_vector.index(f)][k]
            sigmoid = 1.0 / (1.0 + np.exp(-1.0*z))
            if sigmoid >= res:
                c = k
                res = sigmoid
        # return the highest (== the most probable)
        return self.classes[c]
