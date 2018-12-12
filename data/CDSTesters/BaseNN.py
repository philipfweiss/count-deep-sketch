from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import re
import random
import numpy as np
import spacy
from GBDataset import GBDataset
from CDSTester import *


spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')

i = 0

class NNBaseModel:
    def __init__(self):
        self.model = Sequential()
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


        x_train_set = np.array([self.featureExtractor(*item) for item in train_set[0]])
        Y = np.array(train_set[1])

        print(len(x_train_set), len(Y))
        self.model.fit(x_train_set, Y, batch_size=10, verbose=2)
        self.writeTableToFile()

    def writeTableToFile(self):
        np.save(self.file, self.model)

    def generateW2V(self):
        bigram_transformer = Phrases(common_texts)
        print(bigram_transformer)
        w2v = Word2Vec(bigram_transformer[common_texts], size=100, min_count=1)
        print(w2v)
        return w2v

    def featureExtractor(self, item, state, hash, w, d, in_hh):
        # return self.nlpfeatures(item)
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
        in_hh = 1 if in_hh else 0
        feature = [countDiff, countVar, countMean, countSum, numberOfCharsInQuery, in_hh] + self.nlpfeatures(item)
        if self.i % 50 == 0:
            print(self.i)
        self.i+=1
        return feature


    def nlpfeatures(self, item):
        doc = nlp(item.decode('utf8'))
        features = [-1,-1, -1, -1]
        for token in doc:
            # print(token.text)
            features[0] = 1.0 if token.is_stop else 0.
            features[1] = 1.0 if token.is_alpha else 0.
            features[2] = 1.0 if dictionary.check(token) else 0.
            features[3] = len(token)
        # print(doc, features)
        return features

        ## Length
        ## Part of speech of word i
        ## English frequency
        ## isStop
        ## isAlpha
        ##


    def make_prediction(self, observation):
        a = self.model.predict(np.array(observation).reshape(1, -1))[0]
        return a


