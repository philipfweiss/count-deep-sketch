import csv
import numpy

TWITTER_FILENAME = "training.1600000.processed.noemoticon.csv"


# https://stackoverflow.com/questions/4315506/load-csv-into-2d-matrix-with-numpy-for-plotting
reader = csv.reader(open(TWITTER_FILENAME, "r"), quotechar='"', delimiter=",")
for line in reader:
    print(line)
x = list(reader)
result = numpy.array(x).astype("float")

print(result)