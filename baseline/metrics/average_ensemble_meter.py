import numpy as np

class AverageEnsembleMeter:
    def __init__(self, number_of_values):
        self._number_of_values = number_of_values
        self.reset()

    def reset(self):
        self.values = np.zeros((self._number_of_values,))
        self.sums = np.zeros((self._number_of_values,))
        self.counts = 0
        self.averages = np.zeros((self._number_of_values,))
        self.avg = 0

    def update(self, values, n=1):
        if self.counts == 0:
            self.values = values
        else:
            self.values = np.vstack((values, self.values))

        self.sums = np.sum(self.values, axis=0)
        self.counts += n
        self.averages = np.mean(self.values, axis=0)
        self.avg = np.mean(self.averages)