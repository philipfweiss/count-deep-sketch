import os
import sys
import time
import pickle
import random
import xxhash
from numpy.random import choice
import math
import collections
from collections import namedtuple
import matplotlib.pyplot as plt
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../../lib")
from CountMinSketch import CountMinSketch
from Oracle import Oracle
import enchant
import spacy

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')

dictionary = enchant.Dict("en_US")

class CDSTester:
    def __init__(self, model, dataset, n_gram, cmsParams):
        self.model = model
        self.n_gram = n_gram
        self.dataset = dataset
        self.cmsParams = cmsParams
        self.cms = CountMinSketch(*self.cmsParams, hash=self._hash, biasFunc=self.standardBias)
        self.oracle = Oracle()
        self.item_count = 0
        self.hh = None
        self.train, self.test, self.validation = [[], []], [[], []], [[], []]

        eps, delt = cmsParams

        self.heavyHitterCMSs = [
            Oracle(),
            CountMinSketch(1.02 * eps, delt, hash=self._hash, biasFunc=self.standardBias)
        ]

    def importDataset(self, recompute=False):
        name = self.dataset.getName() + "-" + str(self.n_gram)
        METADATA_FILENAME = "storage/%s-METADATA" % name
        CMS_FILENAME = "storage/%s-CMS" % name
        ORACLE_FILENAME = "storage/%s-ORACLE" % name
        CMS_FILENAMEreg = "storage/%s-CMSHH" % name
        CMS_FILENAMEhh = "storage/%s-CMSREG" % name
        i = 0
        hh = collections.defaultdict(int)
        if recompute:


            ## Generate heavy hitters
            for item in self.dataset.getGenerator(self.n_gram):
                if i < 500000:
                    hh[item] += 1
                    i += 1
                    if i == 499999:
                        size = len(hh.keys()) / 30
                        items = sorted([(v, k) for k, v in hh.items()])[-1-size:-1]
                        self.hh = { k for (_, k) in items}
                        print(self.hh)
                        print("---")
                        break

            start_time = time.time()
            print('starting')

            for item in self.dataset.getGenerator(self.n_gram):
                self.cms.record(item)
                if item and item in self.hh:
                    self.heavyHitterCMSs[0].record(item)
                else:
                    self.heavyHitterCMSs[1].record(item)

                self.oracle.record(item)
                self.item_count += 1
            elapsed_time = time.time() - start_time
            print("Total items: {:,}".format(self.item_count))
            print("Elapsed time: {}".format(hms_string(elapsed_time)))

            self.heavyHitterCMSs[0].writeToFile(CMS_FILENAMEhh)
            self.heavyHitterCMSs[1].writeTableToFile(CMS_FILENAMEreg)

            self.cms.writeTableToFile(CMS_FILENAME)
            self.oracle.writeToFile(ORACLE_FILENAME)
            f = open(METADATA_FILENAME, 'w+')
            f.write(pickle.dumps((self.item_count)))
            f.close()
        else:
            f = open(METADATA_FILENAME, 'r')
            self.item_count = pickle.loads(f.read())
            f.close()
            self.cms.readTableFromFrile(CMS_FILENAME)
            self.heavyHitterCMSs[0].readFromFrile(CMS_FILENAMEhh)
            self.hh = set(self.heavyHitterCMSs[0].freq.keys())
            self.heavyHitterCMSs[1].readTableFromFrile(CMS_FILENAMEreg)

            self.oracle.readFromFrile(ORACLE_FILENAME)

    def partitionData(self):
        if self.train[0]: return
        TRAIN_PROP, TEST_PROP, VALIDATE_PROP = 0.4, 0, 0.05
        print("partitioning data")
        oracleKeys = self.oracle.freq.keys()
        # for idx, item in enumerate(random.sample(oracleKeys, int(float(len(oracleKeys)) * (TRAIN_PROP + TEST_PROP + VALIDATE_PROP)))):

        for idx, item in enumerate(self.getKeysProportional(
            int(float(len(oracleKeys)) * (TRAIN_PROP + TEST_PROP + VALIDATE_PROP))
        )):
            if idx % 100 == 0: print(idx)

            trueFreq = self.oracle.estimate(item)
            keycounts = sum(self.oracle.freq.values())
            y = (self.heavyHitterCMSs[1].estimateIgnoringBias(item) - trueFreq)

            x = (item, self.cms.table, self.cms.hash, self.cms.w, self.cms.d, item in self.hh)
            rand = random.uniform(0, 1)
            if rand < TRAIN_PROP:
                if item in self.hh: continue
                self.train[0].append(x)
                self.train[1].append(y)
            elif rand < TRAIN_PROP + TEST_PROP:
                self.test[0].append(x)
                self.test[1].append(y)
            elif rand < TRAIN_PROP + TEST_PROP + VALIDATE_PROP:
                self.validation[0] += [x]
                self.validation[1] += [y]
        print("data partitioned")
        print("Trainset", len(self.train[0]))
        print("Validation Set", len(self.validation[0]))

    def getKeysProportional(self, numToGet):
        print("num", numToGet)
        kv = [(k, v) for k, v in self.oracle.freq.items()]
        valsum = float(sum(v for v in self.oracle.freq.values()))
        keys = [k for (k, _) in kv]
        vals = [v / valsum for (_, v) in kv]

        draw = choice(keys, numToGet, p=vals)
        for i in draw: yield i

    def trainModel(self):
        self.partitionData()
        print("training")
        self.model.train_model(self.train)
        print("training complete")
    #
    # def evaluateModel(self):
    #     self.partitionData()
    #     self.model.evaluate(self.test)

    def evaluateResults(self):
        self.partitionData()
        print('beginning evaluation')
        numWords = 1
        runningTotalCount = 1
        epsTimesCount = self.item_count * self.cmsParams[0]
        numErrors = 0
        smartErrorSummed, naiveErrorSummed, avsum = 0, 0, 0
        naiveErr = []
        smartErr = []
        avErr = []
        xAxis = []
        numItems = 0
        badEstAv = 10

        items = len(self.validation[0])
        print(self.hh)
        for i, (x, y) in enumerate(zip(self.validation[0], self.validation[1])):
            item = x[0]
            if item != '' and item is not None:
                numItems += 1
                oracleEstimate = self.oracle.estimate(item)
                runningTotalCount += oracleEstimate

                if item in self.hh:
                    estimate = self.heavyHitterCMSs[0].estimate(item)
                else:
                    estimate = self.heavyHitterCMSs[1].estimate(item)


                badEst = self.cms.estimateIgnoringBias(item)
                badEstAv = (badEstAv + (badEst - oracleEstimate)) / 2.0


                print("_____")
                print("ITEM: ", item)
                print("Smart error", estimate - oracleEstimate)
                print("Naive error", badEst - oracleEstimate)
                print("Average Error", badEstAv - oracleEstimate)
                # print("_____")

                # print(abs(estimate - oracleEstimate), abs(badEst - oracleEstimate))
                smartErrorSummed += abs(estimate - oracleEstimate)
                naiveErrorSummed += abs(badEst - oracleEstimate)
                avsum += badEstAv
                naiveErr.append(naiveErrorSummed)
                smartErr.append(smartErrorSummed)
                avErr.append(avsum + 30)

                xAxis.append(i)

                if estimate >= oracleEstimate + epsTimesCount:
                    numErrors += 1
                if numItems % 500 == 0:
                    print(item, estimate, oracleEstimate)
                    print("number of words so far ", numItems, " number of incorrect words", numErrors, " error rate ", 1.0 * numErrors / numWords)

        ### Graph Results ###
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        ax1.legend(loc=2)
        plt.xlabel("Number of words queried")
        plt.ylabel("Cumulative Error (Absolute)")
        plt.title("Cumulative Error of Count-Min-Sketch With and Without Learned Bias ")
        ax1.plot(xAxis, naiveErr,label="Cumulative Error without learned bias", color="darkcyan")
        ax1.plot(xAxis, smartErr,label="Cumulative Error with learned bias", color="crimson")
        # ax1.plot(xAxis, avErr ,label="Cum. Error with average bias", color="black")

        ax1.legend(loc=2)
        plt.show()
        ### Graph Results ###


        print(numErrors, numWords)
        print("Total Number of Distinct Words: {:,}".format(numWords))
        print("Total Error Rate: {:,}".format(1.0 * numErrors / numWords))
        print("Expected Error Rate: {:,}".format(self.cmsParams[1]))

    def standardBias(self, item, state, hash, w, d):
        features = self.model.featureExtractor(item, state, hash, w, d, item in self.hh)
        # print(features, self.model.make_prediction(features))

        return self.model.make_prediction(features)

    def _hash(self, w, strng, idx):
        h = str(hash((strng, idx)))
        a = xxhash.xxh64(h.encode('utf-8'))
        return int(a.hexdigest(), 16) % w

class Dataset:
    def getGenerator(self):
        raise Exception("Needs to be ovverriden")

    def getName(self):
        return "default"


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)
