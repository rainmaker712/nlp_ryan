import numpy as np
import tensorflow as tf
from collections import OrderedDict

def textcnn(input_placeholder, target_placeholder, vocab_size, embedding_dim, filter_sizes, num_filters, is_training=True, keep_prob=0.8, scope='TextCNN'):
    
    # Get 'sequence_length' and 'num_classes'
    sequence_length = input_placeholder.get_shape()[1]
    num_classes = target_placeholder.get_shape()[1]
        
    # Declare 'end_points' which is an ordered dictionary
    end_points = OrderedDict()
    
    # tf.random_uniform_initializer의 형태를 간소화
    random_uniform = lambda minval, maxval: tf.random_uniform_initializer(minval=minval, maxval=maxval)
    
    # tf.truncated_normal_initializer의 형태를 간소화
    trunc_normal = lambda stddev: tf.truncated_normal_initializer(mean=0.0, stddev=stddev)

    # tf.contrib.layers.xavier_initializer의 형태를 간소화
    xavier = tf.contrib.layers.xavier_initializer()

    # tf.contrib.layers.xavier_initializer_conv2d의 형태를 간소화
    xavier_conv = tf.contrib.layers.xavier_initializer_conv2d()

    # tf.constant_initializer의 형태를 간소화
    constant = lambda value: tf.constant_initializer(value=value)
    
    with tf.variable_scope(scope):
        
        end_point = 'Embedding'
        with tf.variable_scope(end_point):
            w_embedding = tf.get_variable(name='w_embedding', shape=[vocab_size, embedding_dim], 
                                          initializer=random_uniform(-1.0, 1.0))
            embedded_chars = tf.nn.embedding_lookup(params=w_embedding, ids=input_placeholder, name='embedded_chars')
            embedded_chars_expanded = tf.expand_dims(input=embedded_chars, axis=-1, name='embedded_chars_expanded')
            end_points[end_point] = w_embedding
        
        pooled_output = []
        for i, filter_size in enumerate(filter_sizes):
            end_point = 'Conv-maxpool-%d' % filter_size
            with tf.variable_scope(end_point):
                filter_shape = [filter_size, embedding_dim, 1, num_filters]
                bias_shape = [num_filters]
                w_conv = tf.get_variable(name='w_conv', shape=filter_shape, initializer=trunc_normal(0.01))
                b_conv = tf.get_variable(name='b_conv', shape=bias_shape, initializer=constant(0.0))
                conv = tf.nn.conv2d(input=embedded_chars_expanded, filter=w_conv, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                activated = tf.nn.relu(features=tf.nn.bias_add(conv, b_conv), name='relu')
                pooled = tf.nn.max_pool(value=activated, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='maxpool')
                pooled_output.append(pooled)
                end_points[end_point] = pooled
        
        end_point = 'Flatten'
        with tf.variable_scope(end_point):
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(values=pooled_output, axis=3, name='concat')
            h_pool_flat = tf.reshape(tensor=h_pool, shape=[-1, num_filters_total], name='flatten')
            end_points[end_point] = h_pool_flat
        
        end_point = 'Fully-connected'
        with tf.variable_scope(end_point):
            dropout = tf.contrib.slim.dropout(h_pool_flat, keep_prob=keep_prob, is_training=is_training, scope='dropout')
            w_fc = tf.get_variable(name='w_fc', shape=[num_filters_total, num_classes], initializer=xavier)
            b_fc = tf.get_variable(name='b_fc', shape=[num_classes], initializer=constant(0.0))
            logits = tf.nn.xw_plus_b(x=dropout, weights=w_fc, biases=b_fc, name='logits')
            end_points[end_point] = logits
    
    return logits, end_points