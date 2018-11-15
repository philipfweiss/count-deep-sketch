import os
import sys
import re
import pickle
import numpy as np
import random
from gensim.test.utils import common_texts
from gensim.models import Word2Vec, Phrases
from sklearn import svm
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../lib")
import hashlib
from CountMinSketch import CountMinSketch
from Oracle import Oracle
# https://github.com/jeffheaton/article-code/blob/master/python/wikipedia/wiki-basic-stream.py
import xml.etree.ElementTree as etree
import time
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
from sklearn.ensemble import RandomForestRegressor
import math
import matplotlib.pyplot as plt

#TODO use nueral network with word2vec to do this



def _hash(w, strng, idx):
    h = str(hash((strng, idx)))
    a = hashlib.md5(h.encode('utf-8'))
    return int(a.hexdigest(), 16) % w

class MLModel:
    def __init__(self):
        self.regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100, n_jobs=-1, verbose=0)
        self.w2v = None
        self.file = "Saved/modelSaved"
        self.i = 0
    def train_model(self, train_set):
        # each X example is (item, state, hash, w, d)
        # assumes [[X examples for training], [Y examples for training]]
        self.w2v = self.generateW2V()
        # print(self.w2v['the'], len(self.w2v))
        print(len(train_set[0]))
        x_train_set = [self.featureExtractor(*ex) for ex in train_set[0]]
        x_train_set = np.array(x_train_set)
        Y = np.array(train_set[1])

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
        print("SCREE")
        return w2v

    def featureExtractor(self, item, state, hash, w, d):
        # features as a part of the state:
            # frequency of the word in the english laungage,
            # total number of things added to the cms
            # the count min
            # the count max
            # the max - min
            # variance of the bucket values
            # mean of the bucket values
            # sum of hash bucket values
            # len(query)
            # len(query.split(" "))
            # w * d and w, d
        # freqInEnglish = random.random() * 5 #TODO worry about this and get it from the internet
        numItemsInCMS = sum(state[0])
        items = item.split(" ")
        w2v = np.zeros((100))
        for i in items:
            if i in self.w2v:
                w2v += self.w2v[i]

        countValues = [state[i][hash(w, item, i)] for i in range(d)]
        countMin = min(countValues)
        countMax = max(countValues)
        countDiff = countMax - countMin
        countVar = np.var(countValues)
        countMean = np.mean(countValues)
        countSum = sum(countValues)
        numberOfCharsInQuery = len(item)
        numberOfWordsInQuery = len(items)
        # return [1]
        feature =  np.concatenate((w2v, [numItemsInCMS, countMax, countDiff, countVar, countMean, countSum, numberOfCharsInQuery, numberOfWordsInQuery, w * d, w, d]), axis=0)
        if self.i % 50 == 0:
            print(self.i)
        self.i += 1
        return feature



    def make_prediction(self, observation):
        a = self.regr.predict(np.array(observation).reshape(1, -1))
        # print("ree", a)
        return a

mlModel = MLModel()

def standardBias(item, state, hash, w, d):
    features = mlModel.featureExtractor(item, state, hash, w, d)
    return mlModel.make_prediction(features)

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def strip_tag_name(t):
    t = elem.tag
    idx = k = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t

training_set = []
testing_set = []

def proccessPage(page):
    n = 3
    splitPage = page.split(' ')
    splitPage = list(map(lambda w: re.sub('[^a-zA-Z]+', '', w), splitPage))
    splitPage = list(filter(lambda w: w != "" and w is not None, splitPage))
    for i in range(int(len(splitPage) / n)):
        words = splitPage[i * n: (i + 1) * n]
        string = " ".join(words)
        cms.record(string)
        oracle.record(string)
        training_set.append(string)

def streamWikipediaData(funcToCall):
    totalCount = 0
    last = 0
    words = 0
    for event, elem in etree.iterparse(FILENAME_WIKI, events=('start', 'end')):
        thous = totalCount % 100
        if thous == 0 and totalCount != last:
            print(totalCount)
            last = totalCount
        if event == 'start':
            if elem.text is not None:
                funcToCall(elem.text)
                totalCount += 1
                words += len(elem.text)
        else:
            elem.clear()

    return totalCount, last, words

def computeWikipediaW2v():

    def wikisentences():
        i = 0
        for event, elem in etree.iterparse(FILENAME_WIKI, events=('start', 'end')):
            if i > 100000: return
            if elem.text:
                sentence = [word for word in elem.text.split(' ')]
                i += 1
                if (i % 500 == 0):
                    print(i)
                yield sentence

    sent = [s for s in wikisentences()]
    model = Word2Vec(sent)
    model.wv.save_word2vec_format('Saved/model.txt', binary=False)

    # splitPage = page.split(' ')
    # splitPage = list(map(lambda w: re.sub('[^a-zA-Z]+', '', w), splitPage))
    # splitPage = list(filter(lambda w: w != "" and w is not None, splitPage))
    # for i in range(int(len(splitPage) / n)):


if __name__ == '__main__':

    EPSILON = 0.00005
    DELTA = 0.0001
    cms = CountMinSketch(EPSILON, DELTA, _hash, standardBias)
    oracle = Oracle()

    FILENAME_WIKI = '/Users/philipweiss/Work/count-deep-sketch/data/enwiki-latest-pages-articles1.xml-p10p30302'
    ENCODING = "utf-8"
    TEST_SET_SIZE = 100000
    CMS_FILE = 'Saved/cmsSaved'
    ORACLE_FILE = 'Saved/oracleSaved'
    METADATA_FILE = 'Saved/metadataSaved'
    W2V_FILE = 'Saved/w2v'
    RECOMPUTE = False
    PORPORTION_TO_TRAIN_ON = 1.0/100.0



    if RECOMPUTE:

        start_time = time.time()
        print('starting')

        totalCount, last, words = streamWikipediaData(proccessPage)
        # computeWikipediaW2v()

        elapsed_time = time.time() - start_time
        print("Total pages: {:,}".format(totalCount))
        print("Total words: {:,}".format(words))
        print("Elapsed time: {}".format(hms_string(elapsed_time)))

        cms.writeTableToFile(CMS_FILE)
        oracle.writeToFile(ORACLE_FILE)
        f = open(METADATA_FILE, 'w+')
        f.write(pickle.dumps((words)))
        f.close()
    else:
        f = open(METADATA_FILE, 'r')
        (totalCount) = pickle.loads(f.read())
        f.close()
        cms.readTableFromFrile(CMS_FILE)
        oracle.readFromFrile(ORACLE_FILE)

    cleaned_train_set = [[], []]
    oracleKeys = oracle.freq.keys()
    i = 0
    for word in random.sample(oracleKeys, int(float(len(oracleKeys)) * PORPORTION_TO_TRAIN_ON)):
        i += 1
        cleaned_train_set[1].append(cms.estimateIgnoringBias(word) - oracle.estimate(word))
        cleaned_train_set[0].append((word, cms.table, cms.hash, cms.w, cms.d))

    mlModel.train_model(cleaned_train_set)
    # pickle.dump(mlModel, 'mlmodel' + str(time.time()))




    #TODO this is the section you touch to get results!
    print('beginning evaluation')
    numWords = 1
    runningTotalCount = 1
    epsTimesCount = totalCount * EPSILON
    numErrors = 0
    smartErrorSummed, naiveErrorSummed = 0, 0
    # print(len(oracleKeys))
    naiveErr = []
    smartErr = []
    x = []
    for i, word in enumerate(oracleKeys):
        if numWords == 400:
            break
        if word != '' and word is not None:
            numWords += 1
            oracleEstimate = oracle.estimate(word)
            runningTotalCount += oracleEstimate
            estimate = cms.estimate(word)
            badEst = cms.estimateIgnoringBias(word)
            print(estimate - oracleEstimate)
            # print(abs(estimate - oracleEstimate), abs(badEst - oracleEstimate))
            smartErrorSummed += abs(estimate - oracleEstimate)
            naiveErrorSummed += abs(estimate - badEst)
            # print(
            #     smartErrorSummed,
            #     naiveErrorSummed
            # )
            naiveErr.append(naiveErrorSummed)
            smartErr.append(smartErrorSummed)
            x.append(i)

            if estimate >= oracleEstimate + epsTimesCount:
                numErrors += 1
            if numWords % 50 == 0:
                print(word, estimate, oracleEstimate)
                print("number of words so far ", numWords, " number of incorrect words", numErrors, " error rate ", 1.0 * numErrors / numWords)

    ### Graph Results ###
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.legend(loc=2)
    plt.xlabel("Number of words queried")
    plt.ylabel("Cum. Error (Absolute)")
    plt.title("Error of Count-Min-Sketch With and Without Learned Bias")
    ax1.plot(x, naiveErr,label="Cum. Error without learned bias", color="darkcyan")
    ax1.plot(x, smartErr,label="Cum. Error with learned bias", color="crimson")
    ax1.legend(loc=2)
    plt.show()
    ### Graph Results ###


    print(numErrors, numWords)
    print("Total Number of Distinct Words: {:,}".format(numWords))
    print("Total Error Rate: {:,}".format(1.0 * numErrors / numWords))
    print("Expected Error Rate: {:,}".format(DELTA))
