import xml.etree.ElementTree as etree
from CDSTester import *
from sklearn.ensemble import RandomForestRegressor
from GloVEc import *
import re
import random
import numpy as np
import spacy
spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')


class GBDataset(Dataset):
    def __init__(self):
        pass

    def proccessPage(self, page, n_gram):
        pass

    def getGenerator(self, n_gram):
        directory = '/Users/philipweiss/Work/count-deep-sketch/data/Gut/Gutenberg/txt'
        for filename in os.listdir(directory):
            with open(directory + "/" + filename) as f:
                if random.uniform(0, 1) < .01:
                    print(filename)
                    for line in f.readlines():
                        stripped = [re.sub(r'[^\w\s]','',s) for s in line.strip().split()]
                        for i in range(len(stripped) - n_gram + 1):
                            yield " ".join(stripped[i:i+n_gram])


    def getName(self):
        return "GBDataset"

class MLModel:
    def __init__(self):
        self.regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100, n_jobs=-1, verbose=0)
        self.w2v = None
        self.file = "storage/GBDataset-model"
        # self.W, self.vocab, self.ivocab = generate()
        self.i = 0

    def train_model(self, train_set):
        # each X example is (item, state, hash, w, d)
        # assumes [[X examples for training], [Y examples for training]]
        # self.w2v = self.generateW2V()
        # print(self.w2v['the'], len(self.w2v))
        totalLen = len(train_set[0])
        print(totalLen)

        x_train_set = np.array([self.featureExtractor(*item) for item in train_set[0]])
        Y = np.array(train_set[1]).reshape(1, -1)

        print(x_train_set, Y)
        self.regr.fit(x_train_set, Y)
        print(self.regr.feature_importances_)
        self.writeTableToFile()

    def writeTableToFile(self):
        np.save(self.file, self.regr)

    def generateW2V(self):
        bigram_transformer = Phrases(common_texts)
        print(bigram_transformer)
        w2v = Word2Vec(bigram_transformer[common_texts], size=100, min_count=1)
        print(w2v)
        return w2v


    # def featureExtractor(self, item, state, hash, w, d):
    #     vec = [0 for i in range(300)]
    #     if item[0] in self.vocab:
    #         vocab = self.vocab[item[0]]
    #         if vocab < len(self.W) and vocab > 0:
    #             vec = self.W[vocab]
    #
    #     numItemsInCMS = sum(state[0])
    #     items = item.split(" ")
    #     countValues = [state[i][hash(w, item, i)] for i in range(d)]
    #     countMin = min(countValues)
    #     countMax = max(countValues)
    #     countDiff = countMax - countMin
    #     countVar = np.var(countValues)
    #     countMean = np.mean(countValues)
    #     countSum = sum(countValues)
    #     numberOfCharsInQuery = len(item)
    #     numberOfWordsInQuery = len(items)
    #     feature = [countDiff, countVar, countMean, countSum, numberOfCharsInQuery]
    #     if self.i % 50 == 0:
    #         print(self.i)
    #     self.i += 1
    #
    #     return np.concatenate([vec, feature])



    def featureExtractor(self, item, state, hash, w, d):
        numItemsInCMS = sum(state[0])
        items = item.split(" ")
        countValues = [state[i][hash(w, item, i)] for i in range(d)]
        countMin = min(countValues)
        countMax = max(countValues)
        countDiff = countMax - countMin
        countVar = np.var(countValues)
        countMean = np.mean(countValues)
        countSum = sum(countValues)
        numberOfCharsInQuery = len(item)
        numberOfWordsInQuery = len(items)
        feature = [countDiff, countVar, countMean, countSum, numberOfCharsInQuery] + self.nlpfeatures(item)
        return feature


    def nlpfeatures(self, item):
        doc = nlp(item.decode('utf8'))
        features = [0,0]
        for token in doc:
            features[0] = 1.0 if token.is_stop else 0.
            features[1] = 1.0 if token.is_alpha else 0.
        return features

        ## Length
        ## Part of speech of word i
        ## English frequency
        ## isStop
        ## isAlpha
        ##


    def make_prediction(self, observation):
        a = self.regr.predict(np.array(observation).reshape(1, -1))
        return a

EPSILON, DELTA = 0.00005, 0.001
tester = CDSTester(
    MLModel(),
    GBDataset(),
    1,
    (EPSILON, DELTA)
)

tester.importDataset(recompute=False)
tester.trainModel()
# tester.evaluateModel()
tester.evaluateResults()
