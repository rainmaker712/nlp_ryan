import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#%matplotlib inline

import sys

tf.reset_default_graph()

t = np.array([float(i)*0.01 for i in range(10000+1)])
sin = np.sin(t[:-1])
sin_next = np.sin(t[1:])

print_summaries = 3
print(sin[:print_summaries], sin[-print_summaries:])
print(sin_next[:print_summaries], sin_next[-print_summaries:])

time_step = 100
reshaped_sin = np.reshape(sin, [-1, time_step, 1])
reshaped_sin_next = np.reshape(sin_next, [-1, 1])

print(reshaped_sin)
print(reshaped_sin_next)

signal = tf.placeholder(tf.float32, [None, time_step, 1])
signal_next = tf.placeholder(tf.float32, [None, 1])

inputs = tf.unstack(signal, axis=1)
for i, _input in enumerate(inputs):
    print('unstacked input {} shape {}'.format(i, _input.shape))

state_size = 10
rnn_cell = tf.nn.rnn_cell.LSTMCell(state_size)
#rnn_cell = tf.nn.rnn_cell.GRUCell(state_size)
outputs, state = tf.nn.static_rnn(rnn_cell, inputs, dtype=tf.float32)

for i, _output in enumerate(outputs):
    print('rnn output {} shape {}'.format(i, _output.shape))

reshaped_outputs = tf.reshape(tf.stack(outputs, axis=1), [-1, state_size])

print('stacked and reshaped rnn output {}'.format(reshaped_outputs.shape))

out = tf.layers.dense(reshaped_outputs, 1, use_bias=False)

print('output {}'.format(out.shape))

loss = tf.losses.mean_squared_error(signal_next, out)
train_op = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, _loss = sess.run([train_op, loss], feed_dict={signal: reshaped_sin, signal_next: reshaped_sin_next})
        if i%100 == 0:
            print('step: {}, loss: {}'.format(i, _loss))
        
    _pred = sess.run(out, feed_dict={signal: reshaped_sin})
    _reshaped_pred = np.reshape(_pred, [-1])
    plt.plot(_reshaped_pred)
    plt.show()

sys.exit(0)

#Dynamic RNN
tf.reset_default_graph()

t = np.array([float(i)*0.01 for i in range(10000+1)])
sin = np.sin(t[:-1])
sin_next = np.sin(t[1:])

print_summaries = 3
print(sin[:print_summaries], sin[-print_summaries:])
print(sin_next[:print_summaries], sin_next[-print_summaries:])


time_step = 5
reshaped_sin = np.reshape(sin, [-1, time_step, 1])
reshaped_sin_next = np.reshape(sin_next, [-1, 1])

signal = tf.placeholder(tf.float32, [None, time_step, 1])
signal_next = tf.placeholder(tf.float32, [None, 1])

rnn_cell = tf.nn.rnn_cell.LSTMCell(10)
outputs, state = tf.nn.dynamic_rnn(rnn_cell, signal, dtype=tf.float32)

reshaped_outputs = tf.reshape(outputs, [-1, 10])
out = tf.layers.dense(reshaped_outputs, 1)

loss = tf.losses.mean_squared_error(signal_next, out)
train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _, _loss = sess.run([train_op, loss], feed_dict={signal: reshaped_sin, signal_next: reshaped_sin_next})
        print('step: {}, loss: {}'.format(i, _loss))
        
    _pred = sess.run(out, feed_dict={signal: reshaped_sin})
    _reshaped_pred = np.reshape(_pred, [-1])
    plt.plot(_reshaped_pred)
    plt.show()

tf.reset_default_graph()

t = np.array([float(i)*0.01 for i in range(10000+1)])
sin = np.sin(t[:-1])
sin_next = np.sin(t[1:])

print_summaries = 3
print(sin[:print_summaries], sin[-print_summaries:])
print(sin_next[:print_summaries], sin_next[-print_summaries:])


time_step = 5
reshaped_sin = np.reshape(sin, [-1, time_step, 1])
reshaped_sin_next = np.reshape(sin_next, [-1, 1])

signal = tf.placeholder(tf.float32, [None, time_step, 1])
signal_next = tf.placeholder(tf.float32, [None, 1])

state_size = 10
lstm_cell_f = tf.nn.rnn_cell.LSTMCell(state_size)
lstm_cell_b = tf.nn.rnn_cell.LSTMCell(state_size)
(output_f, output_b), (state_f, state_b) = tf.nn.bidirectional_dynamic_rnn(
    lstm_cell_f, lstm_cell_b, signal, dtype=tf.float32)


outputs = tf.concat([output_f, output_b], axis=2)
print(outputs.shape)
reshaped_outputs = tf.reshape(outputs, [-1, 2*state_size])
out = tf.layers.dense(reshaped_outputs, 1)

loss = tf.losses.mean_squared_error(signal_next, out)
train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _, _loss = sess.run([train_op, loss], feed_dict={signal: reshaped_sin, signal_next: reshaped_sin_next})
        print('step: {}, loss: {}'.format(i, _loss))
        
    _pred = sess.run(out, feed_dict={signal: reshaped_sin})
    _reshaped_pred = np.reshape(_pred, [-1])
    plt.plot(_reshaped_pred)
    plt.show()