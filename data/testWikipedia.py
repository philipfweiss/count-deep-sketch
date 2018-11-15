import os
import sys
import re
import pickle
import numpy as np
import random
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


#TODO use nueral network with word2vec to do this



def _hash(w, strng, idx):
    h = str(hash((strng, idx)))
    a = hashlib.md5(h.encode('utf-8'))
    return int(a.hexdigest(), 16) % w

def featureExtractor(item, state, hash, w, d):
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
    freqInEnglish = random.random() * 5 #TODO worry about this and get it from the internet
    numItemsInCMS = sum(state[0])
    countValues = [state[i][hash(w, item, i)] for i in range(d)]
    countMin = min(countValues)
    countMax = max(countValues)
    countDiff = countMax - countMin
    countVar = np.var(countValues)
    countMean = np.mean(countValues)
    countSum = sum(countValues)
    numberOfCharsInQuery = len(item)
    numberOfWordsInQuery = len(item.split(" "))
    return [freqInEnglish, numItemsInCMS, countMin, countMax, countDiff, countVar, countMean, countSum, numberOfCharsInQuery, numberOfWordsInQuery, w * d, w, d]

class MLModel:
    def __init__(self):
        self.regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100, n_jobs=-1, verbose=1)

    def train_model(self, train_set):
        # each X example is (item, state, hash, w, d)
        # assumes [[X examples for training], [Y examples for training]]
        print('training model')
        x_train_set = [featureExtractor(*ex) for ex in train_set[0]]
        x_train_set = np.array(x_train_set)
        print(x_train_set)
        print(train_set[1])
        Y = np.array(train_set[1])
        self.regr.fit(x_train_set, Y)
        print(self.regr.feature_importances_)
        print('training completed')

    def make_prediction(self, observation):
        return self.regr.predict(observation)

mlModel = MLModel()

def standardBias(item, state, hash, w, d):
    features = featureExtractor(item, state, hash, w, d)
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

if __name__ == '__main__':

    EPSILON = 0.005
    DELTA = 0.00001
    cms = CountMinSketch(EPSILON, DELTA, _hash, standardBias)
    oracle = Oracle()

    FILENAME_WIKI = 'enwiki-latest-pages-articles1.xml-p10p30302'
    ENCODING = "utf-8"
    TEST_SET_SIZE = 100000
    CMS_FILE = 'cmsSaved'
    ORACLE_FILE = 'oracleSaved'
    METADATA_FILE = 'metadataSaved'
    RECOMPUTE = False
    PORPORTION_TO_TRAIN_ON = 1.0/100000.0

    if RECOMPUTE:

        start_time = time.time()
        print('starting')

        totalCount, last, words = streamWikipediaData(proccessPage)

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

    numWords = 1
    runningTotalCount = 1
    epsTimesCount = totalCount * EPSILON
    numErrors = 0
    errorSummedRate = 0
    for word in oracleKeys:
        if word != '' and word is not None:
            numWords += 1
            oracleEstimate = oracle.estimate(word)
            runningTotalCount += oracleEstimate
            estimate = cms.estimate(word)
            errorSummedRate += estimate - oracleEstimate
            if estimate >= oracleEstimate + epsTimesCount:
                numErrors += 1
        if numWords % 1000 == 0:
            print("number of words so far ", numWords, " number of incorrect words", numErrors, " error rate ", 1.0 * numErrors / numWords)

    print(numErrors, numWords)
    print("Total Number of Distinct Words: {:,}".format(numWords))
    print("Total Error Rate: {:,}".format(1.0 * numErrors / numWords))
    print("Expected Error Rate: {:,}".format(DELTA))
