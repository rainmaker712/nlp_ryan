#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:39:43 2017

@author: naver

https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/1-seq2seq.ipynb
"""

#Change directory

import os
os.getcwd()
os.chdir('/Users/naver/nlp_ryan/Chatbot')

#Vocab

x = [[5, 7, 8], [6, 3], [3], [1]]

import helpers

xt, xlen = helpers.batch(x)

#Build Model

import numpy as np
import tensorflow as tf

tf.reset_default_graph()
sess = tf.InteractiveSession()

#Model Input and Output

PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

#shape [max_time, batch_size]
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

decoder_inputs = tf.placeholder(shape=(None,None), dtype=tf.int32, name='decoder_inputs')

#None으로 넣었을 때, 제한이 있음.
#Feed values same batch size
#Decoder inputs and output have same decoder_max_time


#Embeddings

# encoder and decoder RNN = [max_time, batch_size, input_embedding_size]
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_inputs_embedded,
        dtype=tf.float32, time_major=True,
        )

del encoder_outputs
"""
seq2seq에서 필요 없기 때문에 제거. Encoder가 rollout될 때, LSTM의 hidden cell만 필요

LSTM은 tensor의 튜플형을 저장함
encoder_final_state.h: Activation of hidden layer of LSTM cell
encoder_final_state.c: final output 

"""

"""
tf.nn.dynamic_rnn
cell - An instance of RNN cell, single tensor
inputs - RNN inputs
initial_state - (optional) An initial state for the RNN. 
If cell.state_size is an integer, this must be a Tensor of appropriate type and shape 
[batch_size, cell.state_size]. If cell.state_size is a tuple, this should be a tuple of tensors having shapes
 [batch_size, s] for s in cell.state_size

"""

#Decoder
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        decoder_cell, decoder_inputs_embedded,
        
        initial_state=encoder_final_state,     
        
        dtype = tf.float32, time_major=True, scope="plain_decoder",
        
        )

#Decoder의 결과물을 사용하여 단어 순서의 분포를 얻음
decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
decoder_prediction = tf.argmax(decoder_logits, 2) #Why 2?

#Optimizer
decoder_logits

"""
RNN output = [max_time, batch_size, hidden_units]
[max_time, batch_size, vocab_size]
vocab_size는 static shape이며, max_time과 batch_sizes는 다이나믹임
"""

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits = decoder_logits,
        )

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdadeltaOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

#Test forward pass
"""
encoder shape is fixed to max
decoder shape

"""
batch_ = [[6], [3,4], [9,8,7]]

batch_, batch_length_ = helpers.batch(batch_)
print('batch_encoded:\n' + str(batch_))

din_, dlen_ = helpers.batch(np.ones(shape=(3,1), dtype=np.int32),
                            max_sequence_length=4)

print('decoder inputs:\n' + str(din_))

pred_ = sess.run(decoder_prediction,
                 feed_dict = {
                         encoder_inputs: batch_,
                         decoder_inputs: din_,
                         })

# Training on the toy task

batch_size = 100

batches = helpers.random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)

print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)

def next_feed():
    batch = next(batches)
    encoder_inputs_, _ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
            [(sequence) + [EOS] for sequence in batch]
            )
    decoder_inputs_, = helpers.batch(
            [[EOS] + (sequence) for sequence in batch]
            )
    return {
            encoder_inputs: encoder_inputs_,
            decoder_inputs: decoder_inputs_,
            decoder_targets: decoder_targets_,
            }
    
loss_track = []

max_batches = 3001
batches_in_epoch = 1000

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()
except KeyboardInterrupt:
    print('training interrupted')