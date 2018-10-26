import numpy
import statistics
import pickle

"""
General Count-Min-Sketch class.
"""
#"The probability that our estimate is off by at most epsilon*||a||_{1} is at least 1 - delta."
class CountMinSketch:
    def __init__(self, epsilon, delta, hash, biasFunc):
        self.w = int(numpy.ceil(numpy.e / epsilon))
        self.d = int(numpy.ceil(numpy.log(1/delta)))
        self.table = numpy.zeros((self.d, self.w))
        self.hash = hash
        self.bias = biasFunc
    

    """
    Records an item being streamed into the CountMinSketch.
    """
    def record(self, item):
        for i in range(self.d):
            loc = self.hash(self.w, item, i)
            self.table[i][loc] += 1

    """
    Estimates the frequency of item.
    """
    def estimate(self, item):
        return (min([
            self.table[i][self.hash(self.w, item, i)] for i in range(self.d)
        ]) - self.bias(
            item, self.table, self.hash)
        )

    ## TODO: Max implement defaultbias https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch
    def estimateRevised(self, item):
        return min(
            self.estimate(item),
            statistics.median(
                [ 
                    ((self.w)*self.table[i][self.hash(self.w, item, i)] - (sum(self.table[0]))/(self.w - 1))
                    for i in range(self.d)
                ]
            ))

    def writeTableToFile(self, filename):
        numpy.save(filename, self.table)

    def readTableFromFrile(self, filename):
        self.table = numpy.load(filename + ".npy")