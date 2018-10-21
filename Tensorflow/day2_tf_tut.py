import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline

import csv

samples = 1000
test_samples = 100
train_dataset = './ffnn_multiclass_dataset/train_dataset.csv'
test_dataset = './ffnn_multiclass_dataset/test_dataset.csv'

up = [i for i in range(10)]
down = [9-i for i in range(10)]
flat = [5 for i in range(10)]

data = []
label = []
for i in range(samples):
    data.append(up)
    data.append(flat)
    data.append(down)
    label.append([0])
    label.append([1])
    label.append([2])
    
for i in range(10):
    print('data: {}, label {}'.format(data[i], label[i]))
    
with open(train_dataset, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(samples-test_samples):
        writer.writerow(label[i] + data[i])
    print('train data is written')
        
with open(test_dataset, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(test_samples):
        writer.writerow(label[i] + data[i])
    print('test data is written')

tf.reset_default_graph()
trainset = tf.contrib.data.TextLineDataset(train_dataset).batch(10)
testset = tf.contrib.data.TextLineDataset(test_dataset).batch(10)

train_itr = trainset.make_one_shot_iterator()
test_itr = testset.make_one_shot_iterator()

train_batch = train_itr.get_next()
test_batch = test_itr.get_next()

decoded_train = tf.decode_csv(train_batch, [[0]]*11)
decoded_test = tf.decode_csv(test_batch, [[0]]*11)

train_label = tf.one_hot(decoded_train[0], depth=3, axis=-1, dtype=tf.float32)
test_label = tf.reshape(decoded_test[0], [-1, 1])

train_data = tf.stack(decoded_train[1:], axis=1)
test_data = tf.stack(decoded_test[1:], axis=1)

test_data = tf.cast(test_data, tf.float32)
train_data = tf.cast(train_data, tf.float32)

print(train_data)
print(test_data)
print(train_label)
print(test_label)

def multi_class_model(x, activation, reuse=False):
    layer1 = tf.layers.dense(x, 10, activation=activation, reuse=reuse, name='layer1')
    layer2 = tf.layers.dense(layer1, 10, activation=activation, reuse=reuse, name='layer2')
    layer3 = tf.layers.dense(layer2, 10, activation=activation, reuse=reuse, name='layer3')
    layer4 = tf.layers.dense(layer3, 10, activation=activation, reuse=reuse, name='layer4')
    return tf.layers.dense(layer4, 3, activation=activation, reuse=reuse, name='layer_out')

train_out = multi_class_model(train_data, tf.nn.sigmoid)
test_out = multi_class_model(test_data, tf.nn.sigmoid, True)

for var in tf.trainable_variables():
    print(var)

loss = tf.losses.softmax_cross_entropy(train_label, train_out)
train_op = tf.train.GradientDescentOptimizer(1e-6).minimize(loss)

pred = tf.nn.softmax(test_out)
accuracy = tf.metrics.accuracy(test_label, tf.argmax(pred, axis=1))

saver = tf.train.Saver()

#추가로 입력값을 저장한다.

with tf.Session() as sess:
 
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    #텐서보드 생성
    summary_writer = tf.summary.FileWriter('tensorboard', tf.get_default_graph())
    if not os.path.exists('tensorboard'):
        os.makedirs('tensorboard')

    # with tf.name_scope('Loss'):
    tf.summary.histogram('Histogram error', loss) #tensorboard 하나라도 있어야 저장됨

    summary_op = tf.summary.merge_all()
       
    for i in range(10000):
        while True:
            try:
                _, _loss, summary = sess.run([train_op, loss, summary_op])
                _acc = sess.run(accuracy)                                   
            
            except tf.errors.OutOfRangeError:
                break
                
        print('epoch: {}, loss: {}, acc: {}'.format(i, _loss, _acc[0]))
        saver.save(sess, './logs/model')
        #TF board
        log_writer = tf.summary.FileWriter('tensorboard')
        log_writer.add_summary(summary, i)
    