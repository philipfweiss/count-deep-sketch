from keras.layers import Dense, Dropout
from BaseNN import NNBaseModel
from GBDataset import GBDataset
from CDSTester import *

class BiggerNN(NNBaseModel):
    def __init__(self):
        NNBaseModel.__init__(self)
        self.model.add(Dense(10, input_dim=10, activation='relu'))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dropout(0.01))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

EPSILON, DELTA = 0.00005, 0.001
tester = CDSTester(
    BiggerNN(),
    GBDataset(),
    1,
    (EPSILON, DELTA)
)


tester.importDataset(recompute=False)
tester.trainModel()
# tester.evaluateModel()
tester.evaluateResults()