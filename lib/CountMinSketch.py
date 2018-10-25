import numpy

"""
General Count-Min-Sketch class.
"""
class CountMinSketch:
    def __init__(self, w, d, hash, biasFunc):
        self.table = numpy.zeros((w, d))
        self.w = w
        self.d = d
        self.hash = hash
        self.bias = biasFunc

    """
    Records an item being streamed into the CountMinSketch.
    """
    def record(self, item):
        for i in range(self.w):
            loc = self.hash(self.d, item, i)
            self.table[i][loc] += 1

    """
    Estimates the frequency of item.
    """
    def estimate(self, item):
        return (min([
            self.table[i][self.hash(self.d, item, i)] for i in range(self.w)
        ]) - self.bias(
            item, self.table, self.hash)
        )

    ## TODO: Max implement defaultbias https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch
    def defaultBias(item):
        pass
