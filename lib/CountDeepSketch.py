import hashlib
from CountMinSketch import CountMinSketch
from Oracle import Oracle

def _hash(d, strng, idx):
    h = str(hash((strng, idx)))
    a = hashlib.sha1(h)
    return int(a.hexdigest(), 16) % d

def standardBias(item, state, hash):
    pass

cms = CountMinSketch(50,50, _hash, (lambda x, y, z: 0))
oracle = Oracle()

for i in range(100):
    cms.record(i)
    oracle.record(i)

print(cms.estimate(3))
print(oracle.estimate(3))

print(cms.estimate(200))
