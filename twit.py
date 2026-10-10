import numpy
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, BisectingKMeans
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext('local[4]','sentiment')

f1 = open("/home/student/Downloads/twitter datasets/positive-words.txt")
# positive = [x for x in f1.readlines()] #This line returns words with \n , so we split it
positive = [x.split('\n')[0] for x in f1.readlines()]
print(positive)
f2 = open("/home/student/Downloads/twitter datasets/negative-words.txt")
negative = [x.split("\n")[0] for x in f2.readlines()]
print(negative)

trainRDD = sc.textFile("/home/student/Downloads/twitter datasets/train.csv")
# trainRDD.foreach(print)
print(trainRDD.take(10))
header = trainRDD.first()
print(header)

train_cleaned=trainRDD.filter(lambda w: w != header).map(lambda w: w) \
                        .map(lambda w: w.split(",,,,,")[0])\
                        .map(lambda w: w.split(',', 2)[-1])

# train_cleaned = trainRDD.filter(lambda w: w != header)
# print(train_cleaned.take(10))
# train_cleaned = train_cleaned.map(lambda x:x.split(',,,,,')[0])
# print(train_cleaned.take(10))
# train_cleaned = train_cleaned.map(lambda w:w.split(',',2)[-1].replace("[^0-9a-zA-Z']",' '))
# print(train_cleaned.take(10))
# print(train_cleaned)
print("tes")
print(train_cleaned.take(10))
transformedRDD =train_cleaned \
        .map(lambda s: [sum([1 if w in positive else 0 for w in s.split(" ")]),
                        sum([1 if w in negative else 0 for w in s.split(" ")])])
# transformedRDD = train_cleaned.map(lambda l: [w for w in l.split(" ")])
# transformedRDD = train_cleaned.map(lambda l: [ 1 if w in positive else 0 for w in l.split(' ')])
# transformedRDD = train_cleaned.map(lambda l:([1 if w in positive else 0 for w in l.split(" ")],
#                                              [1 if w in negative else 0  for w in l. split(' ')]))
# transformedRDD =train_cleaned \
#         .map(lambda s: [sum([1 if w in positive else 0 for w in s.split(" ")]),
#                         sum([1 if w in negative else 0 for w in s.split(" ")])])
print(transformedRDD.take(10))
train_labled_points = transformedRDD.map(lambda w:
                            LabeledPoint(
                                0 if w[0] == max(w)
                                else 1 if w[1] == max(w)
                                else -1
                                ,
                                Vectors.dense(numpy.array(w))
                            ))
print(train_labled_points.take(10))
testDataRDD = sc.textFile("/home/student/Downloads/twitter datasets/test.csv")
header = testDataRDD.first()
test_cleaned = testDataRDD.filter(lambda w: w != header)\
    .map(lambda w: w.split(',')[1])
print(test_cleaned.take(10))
test_transformedRDD =test_cleaned \
        .map(lambda s: [sum([1 if w in positive else 0 for w in s.split(" ")]),
                        sum([1 if w in negative else 0 for w in s.split(" ")])])
test_labled_points = test_transformedRDD.map(lambda w:
                            LabeledPoint(
                                0 if w[0] == max(w)
                                else 1 if w[1] == max(w)
                                else -1
                             ,
                                Vectors.dense(numpy.array(w))
                            ))

# test_labled_points and train_labled_points are the points we use to do k means

trainVector = train_labled_points.map(lambda lp: Vectors.dense(numpy.array([x for x in lp.features])))
print(trainVector.take(10))
testVector = test_labled_points.map(lambda lp: Vectors.dense(numpy.array([x for x in lp.features])))
print(testVector.take(10))

# kmeans

model = KMeans.train(trainVector, k=2, seed=1, maxIterations=2)

print("cluster center :")
for center in model.centers:
    print(center)

outRDD = model.predict(testVector)


def error(point):
    try:
        center = model.centers[model.predict(point)]
        num = numpy.sqrt(sum([x ** 2 for x in (point - center)]))
        return num
    except:
        return numpy.sqrt(sum([x ** 2 for x in point]))


# WSSSE = testVector.map(lambda point: error(point)).reduce(lambda x, y: x + y)
# print()
# print("Within Set Sum of Squared Error = " + str(WSSSE))

#  Confusion matrix
out = test_labled_points.map(lambda lp: [lp.label , float(model.predict(lp.features))])
# print(out.take(20))

# 1.TN , 2.TP, 3.FN, 4.FP
# 1 - negatve
# 0 - positive
# global tp
# tp=0
def confusion_mat(x):
    predicted_out = x[1]
    actual_out = x[0]
    if predicted_out == actual_out:
        if predicted_out == 0:
            return 1  # truepositive
        else:
            return 2  # truenegative
    else:
        if predicted_out == 0:
            return 3  # falsepositive
        else:
            return 4  # falsenegative


# confusion_rdd = out.map(lambda x: tn if x[0] == x[1] == 1 else 2 if x[0] == x[1] == 0 else 3 if(x[0] != x[1] and x[1] == 1)
#                     else 4 if (x[0] != x[1] and x[1] == 0) else 0)
confusion_rdd = out.map(confusion_mat)
#
print(confusion_rdd.take(20))
tp = confusion_rdd.filter(lambda n: n == 1).reduce(lambda x,y:x+y)
tn = confusion_rdd.filter(lambda n: n == 2).count()
fp = confusion_rdd.filter(lambda n: n == 3).count()
fn = confusion_rdd.filter(lambda n: n == 4).count()
print(tp, tn, fp, fn)
print(tp)
bkmModel = BisectingKMeans.train(trainVector, 2, maxIterations=2)

print("bisecting cluster centers :")
for center in bkmModel.centers:
    print(center)

bkmOut = bkmModel.predict(testVector)
out2 = test_labled_points.map(lambda lp: [lp.label,bkmModel.predict(lp.features)])
print(out2.take(10))