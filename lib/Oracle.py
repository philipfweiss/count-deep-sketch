import collections
import pickle

class Oracle:
    def __init__(self):
        self.freq = collections.defaultdict(int)

    def record(self, item):
        self.freq[item] += 1

    def estimate(self, item):
        return self.freq[item]

    def writeToFile(self, filename):
        f = open(filename, 'w+')
        f.write(pickle.dumps(self.freq))

    def readFromFrile(self, filename):
        f = open(filename, 'r')
        self.freq = pickle.loads(f.read())
