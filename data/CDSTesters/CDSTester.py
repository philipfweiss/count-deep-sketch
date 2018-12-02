import os
import sys
import time
import pickle
import random
import xxhash
import math
import matplotlib.pyplot as plt
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../../lib")
import hashlib
from CountMinSketch import CountMinSketch
from Oracle import Oracle

class CDSTester:
    def __init__(self, model, dataset, n_gram, cmsParams):
        self.model = model
        self.n_gram = n_gram
        self.dataset = dataset
        self.cmsParams = cmsParams
        self.cms = CountMinSketch(*self.cmsParams, hash=self._hash, biasFunc=self.standardBias)
        self.oracle = Oracle()
        self.item_count = 0
        self.train, self.test, self.validation = [[], []], [[], []], [[], []]


    def importDataset(self, recompute=False):
        name = self.dataset.getName() + "-" + str(self.n_gram)
        METADATA_FILENAME = "storage/%s-METADATA" % name
        CMS_FILENAME = "storage/%s-CMS" % name
        ORACLE_FILENAME = "storage/%s-ORACLE" % name
        if recompute:
            start_time = time.time()
            print('starting')
            for item in self.dataset.getGenerator(self.n_gram):
                self.cms.record(item)
                self.oracle.record(item)
                self.item_count += 1
            elapsed_time = time.time() - start_time
            print("Total items: {:,}".format(self.item_count))
            print("Elapsed time: {}".format(hms_string(elapsed_time)))

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
            self.oracle.readFromFrile(ORACLE_FILENAME)

    def partitionData(self):
        if self.train[0]: return
        TRAIN_PROP, TEST_PROP, VALIDATE_PROP = 0.5, 0.3, 0.2
        print("partitioning data")
        oracleKeys = self.oracle.freq.keys()
        for idx, item in enumerate(random.sample(oracleKeys, int(float(len(oracleKeys)) * (TRAIN_PROP + TEST_PROP + VALIDATE_PROP)))):
            y = self.cms.estimateIgnoringBias(item) - self.oracle.estimate(item)
            x = (item, self.cms.table, self.cms.hash, self.cms.w, self.cms.d)
            mod = (idx % 10) / 10.0
            if 0 <= mod and mod < TRAIN_PROP:
                self.train[0].append(x)
                self.train[1].append(y)
            elif TRAIN_PROP <= mod and mod < TRAIN_PROP + TEST_PROP:
                self.test[0].append(x)
                self.test[1].append(y)
            else:
                self.validation[0].append(x)
                self.validation[1].append(y)
        print("data partitioned")

    def trainModel(self):
        self.partitionData()
        print("training")
        self.model.train_model(self.train)
        print("training complete")

    def evaluateModel(self):
        self.partitionData()
        self.model.evaluate(self.test)

    def evaluateResults(self):
        self.partitionData()
        print('beginning evaluation')
        numWords = 1
        runningTotalCount = 1
        epsTimesCount = self.item_count * self.cmsParams[0]
        numErrors = 0
        smartErrorSummed, naiveErrorSummed = 0, 0
        naiveErr = []
        smartErr = []
        xAxis = []
        numItems = 0
        for i, (x, y) in enumerate(zip(self.validation[0], self.validation[1])):
            item = x[0]
            if item != '' and item is not None:
                numItems += 1
                oracleEstimate = self.oracle.estimate(item)
                runningTotalCount += oracleEstimate
                estimate = self.cms.estimate(item)
                badEst = self.cms.estimateIgnoringBias(item)
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
        plt.ylabel("Cum. Error (Absolute)")
        plt.title("Error of Count-Min-Sketch With and Without Learned Bias")
        ax1.plot(xAxis, naiveErr,label="Cum. Error without learned bias", color="darkcyan")
        ax1.plot(xAxis, smartErr,label="Cum. Error with learned bias", color="crimson")
        ax1.legend(loc=2)
        plt.show()
        ### Graph Results ###


        print(numErrors, numWords)
        print("Total Number of Distinct Words: {:,}".format(numWords))
        print("Total Error Rate: {:,}".format(1.0 * numErrors / numWords))
        print("Expected Error Rate: {:,}".format(self.cmsParams[1]))

    def standardBias(self, item, state, hash, w, d):
        features = self.model.featureExtractor(item, state, hash, w, d)
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
