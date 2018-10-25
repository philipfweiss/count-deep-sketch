import collections

class Oracle:
    def __init__(self):
        self.freq = collections.defaultdict(int)

    def record(self, item):
        self.freq[item] += 1

    def estimate(self, item):
        return self.freq[item]
