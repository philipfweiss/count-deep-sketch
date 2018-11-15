import os
import sys
import re
import pickle
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


def _hash(w, strng, idx):
    h = str(hash((strng, idx)))
    a = hashlib.md5(h.encode('utf-8'))
    return int(a.hexdigest(), 16) % w



def standardBias(item, state, hash):
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
    pass
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



if __name__ == '__main__':

    EPSILON = 0.005
    DELTA = 0.00001
    cms = CountMinSketch(EPSILON, DELTA, _hash, (lambda x, y, z: 0))
    oracle = Oracle()

    FILENAME_WIKI = 'enwiki-latest-pages-articles1.xml-p10p30302'
    ENCODING = "utf-8"
    TEST_SET_SIZE = 100000
    CMS_FILE = 'cmsSaved'
    ORACLE_FILE = 'oracleSaved'
    METADATA_FILE = 'metadataSaved'
    RECOMPUTE = True

    if RECOMPUTE:
        totalCount = 0
        start_time = time.time()
        last = 0
        words = 0
        print('starting')
        for event, elem in etree.iterparse(FILENAME_WIKI, events=('start', 'end')):
                tname = strip_tag_name(elem.tag)
                thous = totalCount % 100
                if thous == 0 and totalCount != last:
                    print(totalCount)
                    last = totalCount
                if event == 'start':
                    if elem.text is not None:
                        proccessPage(elem.text)
                        totalCount += 1
                        words += len(elem.text)
                else:
                    elem.clear()

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

    numWords = 1
    runningTotalCount = 1
    epsTimesCount = totalCount * EPSILON
    numErrors = 0
    errorSummedRate = 0
    for word in oracle.freq.keys():
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
