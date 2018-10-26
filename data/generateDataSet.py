import random
from gensim.test.utils import datapath
from gensim.corpora import WikiCorpus

def generateIntegerDataset(n=100):
    s = []
    f = open('integerData.csv', 'w+')
    for i in range(n):
        s.append(str(random.expovariate(100)))
    f.write("\n".join(s))
    f.close()
# generateIntegerDataset()

path_to_wiki_dump = datapath("/Users/Lipman/Developer-Files/Classes/Stanford/CS221/count-deep-sketch/data/enwiki-latest-pages-articles1.xml-p10p30302.bz2")
for vec in WikiCorpus(path_to_wiki_dump).getstream():
    print('a')
    print(vec)
    pass


#TODO stream these through
# Word frequencies in wikipedia corpus
# Word getstream()frequencies in arbitrary webpages
# AOL Query Logs (36M queries)
#TODO save oracle to a file after computing
#TODO do assigned part of writeup
#TODO compute baseline and oraclee