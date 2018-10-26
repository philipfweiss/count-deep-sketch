import os
import sys
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
    a = hashlib.sha1(h.encode('utf-8'))
    return int(a.hexdigest(), 16) % w


def standardBias(item, state, hash):
    pass


cms = CountMinSketch(0.001, 0.00001, _hash, (lambda x, y, z: 0))
oracle = Oracle()



FILENAME_WIKI = 'enwiki-latest-pages-articles1.xml-p10p30302'
ENCODING = "utf-8"

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


totalCount = 0
articleCount = 0
redirectCount = 0
templateCount = 0
title = None
start_time = time.time()

for event, elem in etree.iterparse(FILENAME_WIKI, events=('start', 'end')):
        tname = strip_tag_name(elem.tag)

        if event == 'start':
            if tname == 'text' and elem.text is not None:
                for word in elem.text:
                    cms.record(word)
                    oracle.record(word)
                    if (cms.estimate(word) != cms.estimateRevised(word)):
                        print(cms.estimate(word))
                        print(cms.estimateRevised(word))
                        print(oracle.estimate(word))
                        print(" ")
            totalCount += 1
        else:
            elem.clear()

elapsed_time = time.time() - start_time

print("Total pages: {:,}".format(totalCount))
print("Elapsed time: {}".format(hms_string(elapsed_time)))



