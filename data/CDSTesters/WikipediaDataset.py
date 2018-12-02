import xml.etree.ElementTree as etree
from CDSTester import *
from sklearn.ensemble import RandomForestRegressor
import re
import numpy as np
import multiprocessing

class WikipediaDataset(Dataset):
    def __init__(self):
        pass

    def proccessPage(self, page, n_gram):
        n = n_gram
        splitPage = page.split(' ')
        splitPage = list(map(lambda w: re.sub('[^a-zA-Z]+', '', w), splitPage))
        splitPage = list(filter(lambda w: w != "" and w is not None, splitPage))
        for i in range(int(len(splitPage) / n)):
            words = splitPage[i * n: (i + 1) * n]
            string = " ".join(words)
            yield string

    def getGenerator(self, n_gram):
        FILENAME_WIKI = '/Users/philipweiss/Work/count-deep-sketch/data/enwiki-latest-pages-articles1.xml-p10p30302'
        def strip_tag_name(t):
            t = elem.tag
            idx = k = t.rfind("}")
            if idx != -1:
                t = t[idx + 1:]
            return t



        totalCount = 0
        last = 0
        words = 0
        for event, elem in etree.iterparse(FILENAME_WIKI, events=('start', 'end')):
            thous = totalCount % 1000
            if thous == 0 and totalCount != last:
                print(totalCount)
                last = totalCount
            if event == 'start':
                if elem.text is not None:
                    for item in self.proccessPage(elem.text, n_gram):
                        yield item
                    totalCount += 1
                    words += len(elem.text)
            else:
                elem.clear()


    def getName(self):
        return "WikipediaDataset"

class MLModel:
    def __init__(self):
        self.regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100, n_jobs=-1, verbose=0)
        self.w2v = None
        self.file = "Saved/modelSaved"
        self.i = 0
    def train_model(self, train_set):
        # each X example is (item, state, hash, w, d)
        # assumes [[X examples for training], [Y examples for training]]
        # self.w2v = self.generateW2V()
        # print(self.w2v['the'], len(self.w2v))
        totalLen = len(train_set[0])
        print(totalLen)
        p = multiprocessing.Pool(6)
        x_train_set = p.map(self.featureExtractor, train_set[0])

        # x_train_set = [self.featureExtractor(*ex) for ex in train_set[0]]
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
        # w2v = np.zeros((100))
        # for i in items:
        #     if i in self.w2v:
        #         w2v += self.w2v[i]

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
        feature = [numItemsInCMS, countMax, countDiff, countVar, countMean, countSum, numberOfCharsInQuery, numberOfWordsInQuery, w * d, w, d]
        # feature =  np.concatenate((w2v, [numItemsInCMS, countMax, countDiff, countVar, countMean, countSum, numberOfCharsInQuery, numberOfWordsInQuery, w * d, w, d]), axis=0)
        if self.i % 50 == 0:
            print(self.i)
        self.i += 1
        return feature



    def make_prediction(self, observation):
        a = self.regr.predict(np.array(observation).reshape(1, -1))
        # print("ree", a)
        return a

mlModel = MLModel()


EPSILON, DELTA = 0.00005, 0.001
tester = CDSTester(
    MLModel(),
    WikipediaDataset(),
    2,
    (EPSILON, DELTA)
)

tester.importDataset(recompute=False)
tester.trainModel()
tester.evaluateModel()
tester.evaluateResults()
