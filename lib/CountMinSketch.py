import numpy
import hashlib

"""
General Count-Min-Sketch class.
"""
class CountMinSketch:
    def __init__(self, n, d, biasFunc):
        self.table = numpy.zeros((n, d))
        self.n = n
        self.d = d
        self.bias = biasFunc

    """
    Records an item being streamed into the CountMinSketch.
    """
    def record(self, item):
        for i in range(self.n):
            loc = self._hash(item, i)
            self.table[i][loc] += 1

    """
    Estimates the frequency of item.
    """
    def estimate(self, item):
        return min([
            self.table[i][self._hash(item, i)] for i in range(self.n)
        ])

    def _hash(self, strng, idx):
        h = str(hash((strng, idx)))
        a = hashlib.sha1(h)
        return int(a.hexdigest(), 16) % self.d
