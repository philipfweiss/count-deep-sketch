from CDSTester import *

class OneThroughOneThousand(Dataset):
    def __init__(self):
        pass

    def getGenerator(self, n_gram):
        for i in range(1000000): yield str(i) + "" + str(i + 1)

    def getName(self):
        return "1-1000"


class MLModel:
    def train_model(self, train_set):
        print(len(train_set[0]))
    def evaluate(self, test):
        print("eval")
    def featureExtractor(self, item, state, hash, w, d):
        return ()
    def make_prediction(self, features):
        return [1.0]

mlModel = MLModel()


EPSILON, DELTA = 0.00005, 0.001
tester = CDSTester(
    MLModel(),
    OneThroughOneThousand(),
    2,
    (EPSILON, DELTA)
)

tester.importDataset(recompute=True)
tester.trainModel()
tester.evaluateModel()
tester.evaluateResults()
