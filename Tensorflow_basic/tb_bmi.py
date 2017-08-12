import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

csv = pd.read_csv("bmi.csv")
bclass = {"thin": [1,0,0], "normal": [0,1,0], "fat": [0,0,1]}

csv["label_pat"] = csv["label"].apply(lambda x : np.array(bclass[x]))
csv["height"] = csv["height"] / 200
csv["weight"] = csv["weight"] / 100
print(csv)

step=0
rows = csv[step*10:(step+1)*10]
x_pat = rows[["weight","height"]]
y_ans = rows["label_pat"]

csv[0:0]

step = 0
rows = csv[1+step*10:(step+1)*10]
x_pat = rows[["weight","height"]]
y_ans = list(rows["label_pat"])

#플레이스 홀더에 이름 붙이기
x = tf.placeholder(tf.float32, [None, 2], name="x")
y_ = tf.placeholder(tf.float32, [None, 3], name="y_")

sess = tf.Session()
#sess.run(x, feed_dict={x: x_pat})
sess.run(y_, feed_dict={y_: y_ans})

#플레이스 홀더 선언
x    = tf.placeholder(tf.float32, [None, 2], name="x") 
y_ = tf.placeholder(tf.float32, [None, 3], name = 'y_') 

with tf.name_scope('interface') as scope:
    # 변수 선언
    W = tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0), name='W')
    b = tf.Variable(tf.zeros([3]), name='b')
    #softmax 회귀
    with tf.name_scope('softmax') as scope:
        y = tf.nn.softmax(tf.matmul(x,W)+b)

with tf.name_scope('loss') as scope:
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10,1.0)))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

# 정답률 구현
test_csv = csv[15000:20000]
test_pat = test_csv[["weight","height"]]
test_ans = list(test_csv["label_pat"])
# 세션 시작
predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(predict, "float"))

# 학습
sess = tf.Session()
sess.run(tf.initialize_all_variables()) #변수 초기화

for step in range(5000):
    i = (step * 100) % 1000
    rows = csv[1 + i : 1 + i + 100]
    x_pat = rows[["weight","height"]]
    y_ans = list(rows["label_pat"])
    fd = {x: x_pat, y_: y_ans}
    sess.run(train, feed_dict=fd)
    if step % 100 == 0:
        e = sess.run(cross_entropy, feed_dict=fd)
        a = sess.run(acc, feed_dict={x: test_pat, y_: test_ans})
        print("step=", step, "ce=", e, "acc=", a)

# 최종 정답 구하기
test_csv = csv[15000:20000]
test_pat = test_csv[["weight","height"]]
test_ans = list(test_csv["label_pat"])
predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(predict, "float"))
r = sess.run(acc, feed_dict={x: test_pat, y_:test_ans})
print(r)

test = tf.argmax(y, 1)
r = sess.run(test, feed_dict={x: [[70/100, 170/200]]})
print(r)                                                        

tw = tf.summary.FileWriter("log_dir", graph=sess.graph)