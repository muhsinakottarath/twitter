import numpy
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

f1 = open("/home/student/Downloads/twitter datasets/positive-words.txt")
pos = [x.split("\n")[0] for x in f1.readlines()]
# print(pos)

f2 = open("/home/student/Downloads/twitter datasets/negative-words.txt")
neg = [x.split("\n")[0] for x in f2.readlines()]
# print(neg)

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
spark = SparkSession.Builder().master('local')\
    .appName('tsa').getOrCreate()
sc = spark.sparkContext
# sc = SparkContext('local','twittersentimental')

rdd = sc.textFile("/home/student/Downloads/twitter datasets/train.csv")
# rdd.foreach(print)
head = rdd.first()
# print(head)
clean_rdd = rdd.filter(lambda x:x!=head)\
    .map(lambda x: x.split(',', 2)[-1].replace(',,,,,,,,,',''))
print(clean_rdd.take(10))
train_rdd = clean_rdd.map(lambda l:[sum([1 if w in pos else 0 for w in l.split(" ")]),
                          sum([1 if w in neg else 0 for w in l.split(" ")])])
train_points = train_rdd.map(lambda w:LabeledPoint(0 if w[0]==max(w)
                                                   else 1,
                                                   Vectors.dense(numpy.array(w))))
# print(train_rdd.take(10))
trainVector = train_points.map(lambda lp:lp.features)
model = KMeans.train(trainVector,k=2,seed =1,maxIterations=20)
for x in model.centers:
    print(x)

out = model.predict(trainVector) # testVEctor
print(out.take(10))
# train_points.foreach(print)
# trainVector.foreach(print)
# twitter/kmeans.py:25twitter/kmeans.py:25
# [pos,neg]

rdd1 = sc.textFile("/home/student/Downloads/twitter datasets/train.csv")
# rdd.foreach(print)
head = rdd1.first()
# print(head)
clean_rdd1 = rdd1.filter(lambda x:x!=head)\
    .map(lambda x: x.split(',', 2)[-1])
print(clean_rdd1.take(10))
train_rdd = clean_rdd.map(lambda l: [sum([1 if w in pos else 0 for w in l.split(" ")]),
                          sum([1 if w in neg else 0 for w in l.split(" ")])])

test_rdd = clean_rdd.map(lambda l: [sum([1 if w in pos else 0 for w in l.split(" ")]),
                                    sum([1 if w in neg else 0 for w in l.split(" ")])])
train_points = train_rdd.map(lambda w: LabeledPoint(0 if w[0] == max(w)
                                                    else 1,
                                                    Vectors.dense(numpy.array(w))))



# test_points = test_rdd1.map(lambda w:LabeledPoint(0 if w[0]==max(w)
#                              else 1,
#                              Vectors.dense(numpy.array(w))))
# testVectors = test_points.map(lambda lp:lp.features)
# out = model.predict(testVectors)
# print(out.take(10))
# test_label=test_points.map(lambda l:l.label)
# train_label=test_label.zipWithIndex().map(lambda x:(x[1],x(0)))
# out=out.zipWithIndex().map(lambda x:(x[1],x[0]))
# ful=out.join(test_label)
# print(ful.take(10))