import tensorflow as tf
import sys
print(tf.__version__)
print(sys.version)
#print(tf.__file__)
a=tf.random.uniform([2,3],0,1)
print(a)
print(type(a))