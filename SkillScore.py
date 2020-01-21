import pandas as pd
import numpy as np

class ClassificationScores:

    def __init__(self, conf_dt):
        self.conf_dt = conf_dt


    def calc_entropy(self, X, N):

        entropy = -1*(np.sum(X*np.log(X))

        return entropy


    def mutual_info(self):
        #conf_dt is the confidence dataset
        conf_matrix = self.conf_dt.values
        #Sum of true inputs (sum over the rows)
        x_i = np.sum(conf_matrix, axis = 0)
        y_i = np.sum(conf_matrix, axis=1)
        N = np.sum(np.sum(conf_matrix))

        X = x_i/N
        Y = y_i/N
        Z = conf_matrix/N

        H_X = calc_entropy(X)
        H_Y = calc_entropy(Y)
        H_Z = calc_entropy(Z)

        mutual_info = -

        return mutual_info


