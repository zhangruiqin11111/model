# coding=utf-8
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("c01")
sc = SparkContext(conf=conf)
print(sc.version)

