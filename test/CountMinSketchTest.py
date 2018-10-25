import sys
sys.path.insert(0, '../lib')

from CountMinSketch import CountMinSketch
from Oracle import Oracle

cms = CountMinSketch(50,60, lambda x: 0)
oracle = Oracle()

for i in range(300):
    print(i)
    for j in range(i):
        cms.record(i)
        oracle.record(i)

for i in range(300):
    if (cms.estimate(i) != oracle.estimate(i)):
        print("cms: %d - oracle: %d" % (cms.estimate(i), oracle.estimate(i)))
    # print("cms: %d - oracle: %d" % (cms.estimate(i), oracle.estimate(i)))
