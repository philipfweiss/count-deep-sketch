import os
import sys
import re
import collections
import Queue
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../lib")
import hashlib
from CountMinSketch import CountMinSketch
from Oracle import Oracle
# https://github.com/jeffheaton/article-code/blob/master/python/wikipedia/wiki-basic-stream.py
import xml.etree.ElementTree as etree
import time
from multiprocessing.dummy import Pool
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)


def _hash(w, strng, idx):
    h = str(hash((strng, idx)))
    a = hashlib.sha1(h.encode('utf-8'))
    return int(a.hexdigest(), 16) % w


def standardBias(item, state, hash):
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
    splitPage = page.split(' ')
    for word in splitPage:
        word = re.sub('[^a-zA-Z]+', '', word)
        cms.record(word)
        oracle.record(word)


if __name__ == '__main__':

    cms = CountMinSketch(0.001, 0.00001, _hash, (lambda x, y, z: 0))
    oracle = Oracle()

    FILENAME_WIKI = 'enwiki-latest-pages-articles1.xml-p10p30302'
    ENCODING = "utf-8"
    TEST_SET_SIZE = 1000
    CMS_FILE = 'cmsSaved'
    ORACLE_FILE = 'oracleSaved'
    RECOMPUTE = False
    EPSILON = 150

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
                    if tname == 'text' and elem.text is not None:
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
    else:
        cms.readTableFromFrile(CMS_FILE)
        oracle.readFromFrile(ORACLE_FILE)

    totalCount = len(oracle.freq.keys())

    numWords = 0
    wordsDifferent = 0
    errorRate = []
    for word in oracle.freq.keys():
        numWords += 1
        if numWords % 1000 == 0:
            print("number of words so far ", numWords, " num errors ", wordsDifferent, " error rate ", 1.0*wordsDifferent/numWords)
        error = abs(cms.estimateRevised(word) - oracle.estimate(word))
        if (error > EPSILON):
            errorRate.append(error)
            wordsDifferent += 1

    print("Average Error: {:,}".format(sum(errorRate)/len(errorRate)))
    print("Total Number of Words: {:,}".format(numWords))
    print("Number of Words that are wrong: {:,}".format(numWords))
    print("Error Rage: {:,}".format(1.0 * wordsDifferent/numWords))
