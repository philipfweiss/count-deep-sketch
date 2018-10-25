
from CountMinSketch import CountMinSketch
from Oracle import Oracle

cms = CountMinSketch(50,50, lambda x: 0)
oracle = Oracle()

for i in range(100):
    cms.record(i)
    oracle.record(i)

print(cms.estimate(3))
print(oracle.estimate(3))

print(cms.estimate(200))
