import random

def generateIntegerDataset(n=100):
    s = []
    f = open('integerData.csv', 'w+')
    for i in range(n):
        s.append(str(random.expovariate(100)))
    f.write("\n".join(s))
    f.close()
generateIntegerDataset()


#TODO stream these through
# Word frequencies in wikipedia corpus
# Word frequencies in arbitrary webpages
# AOL Query Logs (36M queries)
#TODO save oracle to a file after computing