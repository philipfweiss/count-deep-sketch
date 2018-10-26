import hashlib
from CountMinSketch import CountMinSketch
from Oracle import Oracle


def _hash(w, strng, idx):
    h = str(hash((strng, idx)))
    a = hashlib.sha1(h.encode('utf-8'))
    return int(a.hexdigest(), 16) % w


def standardBias(item, state, hash):
    pass


cms = CountMinSketch(0.001, 0.00001, _hash, (lambda x, y, z: 0))
oracle = Oracle()

for i in range(50):
    for j in range(i):
        cms.record(i)
        oracle.record(i)
for i in range(50):
    if (cms.estimate(i) != cms.estimateRevised(i)):
        print(cms.estimate(i))
        print(cms.estimateRevised(i))
        print(oracle.estimate(i))
        print(" ")
# print(cms.estimate(200))
