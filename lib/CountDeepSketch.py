from CountMinSketch import CountMinSketch

cms = CountMinSketch(50,60, lambda x: 0)
for i in range(100):
    cms.record(i)

print(cms.estimate(200))
