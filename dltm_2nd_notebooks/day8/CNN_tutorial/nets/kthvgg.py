import tensorflow as tf
from collections import OrderedDict

def vgg(input_placeholder, num_classes, keep_prob=0.7, scope='KTH_VGG'):
    # Inputs에서 [None, height, width, channels] 모양의 텐서가 placeholder로 들어온다고 가정
    height = input_placeholder.get_shape()[1]
    width = input_placeholder.get_shape()[2]
    channels = input_placeholder.get_shape()[3]

    # tf.truncated_normal_initializer의 형태를 간소화
    trunc_normal = lambda stddev: tf.truncated_normal_initializer(mean=0.0, stddev=stddev)

    # tf.contrib.layers.xavier_initializer의 형태를 간소화
    xavier = tf.contrib.layers.xavier_initializer()

    # tf.contrib.layers.xavier_initializer_conv2d의 형태를 간소화
    xavier_conv = tf.contrib.layers.xavier_initializer_conv2d()

    # tf.constant_initializer의 형태를 간소화
    constant = lambda value: tf.constant_initializer(value=value)

    # Define "end_points"
    end_points = OrderedDict()

    with tf.variable_scope(scope):
        
        # Conv_3x3_0
        inputs = input_placeholder
        in_channels = channels
        out_channels = 32
        end_point = 'Conv_3x3_0'
        with tf.variable_scope(end_point):
            W_conv0 = tf.get_variable("weights", [3, 3, in_channels, out_channels], initializer=xavier_conv)
            b_conv0 = tf.get_variable("biases", [out_channels], initializer=constant(0.0))
            with tf.name_scope('conv'):
                conv0 = tf.nn.conv2d(inputs, W_conv0, strides=[1, 1, 1, 1], padding='SAME') + b_conv0
                conv0 = tf.nn.relu(conv0)
            end_points[end_point] = conv0
        
        # MaxPool_2x2_0
        inputs = end_points[end_point]
        end_point = 'MaxPool_2x2_0'
        with tf.variable_scope(end_point):
            pool0 = tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="maxpool")
            end_points[end_point] = pool0
        
        # Conv_3x3_1
        inputs = end_points[end_point]
        in_channels = out_channels
        out_channels = 64
        end_point = 'Conv_3x3_1'
        with tf.variable_scope(end_point):
            W_conv1 = tf.get_variable("weights", [3, 3, in_channels, out_channels], initializer=xavier_conv)
            b_conv1 = tf.get_variable("biases", [out_channels], initializer=constant(0.0))
            with tf.name_scope('conv'):
                conv1 = tf.nn.conv2d(inputs, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
                conv1 = tf.nn.relu(conv1)
            end_points[end_point] = conv1
        
        # MaxPool_2x2_1
        inputs = end_points[end_point]
        end_point = 'MaxPool_2x2_1'
        with tf.variable_scope(end_point):
            pool1 = tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="maxpool")
            end_points[end_point] = pool1
        
        # Conv_3x3_2
        inputs = end_points[end_point]
        in_channels = out_channels
        out_channels = 128
        end_point = 'Conv_3x3_2'
        with tf.variable_scope(end_point):
            W_conv2 = tf.get_variable("weights", [3, 3, in_channels, out_channels], initializer=xavier_conv)
            b_conv2 = tf.get_variable("biases", [out_channels], initializer=constant(0.0))
            with tf.name_scope('conv'):
                conv2 = tf.nn.conv2d(inputs, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
                conv2 = tf.nn.relu(conv2)
            end_points[end_point] = conv2
        
        # MaxPool_2x2_2
        inputs = end_points[end_point]
        end_point = 'MaxPool_2x2_2'
        with tf.variable_scope(end_point):
            pool2 = tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="maxpool")
            end_points[end_point] = pool2
        
        # Flatten
        inputs = end_points[end_point]
        shape = inputs.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        end_point = 'Flatten'
        with tf.variable_scope(end_point):
            flatten = tf.reshape(inputs, [-1, dim], name='reshape')
            end_points[end_point] = flatten
        
        # FC_0
        inputs = end_points[end_point]
        in_dim = dim
        out_dim = 2350
        end_point = 'FC_0'
        with tf.variable_scope(end_point):
            W_fc0 = tf.get_variable("weights", [in_dim, out_dim], initializer=xavier)
            b_fc0 = tf.get_variable("biases", [out_dim], initializer=constant(0.0))
            dropped = tf.nn.dropout(tf.matmul(inputs, W_fc0) + b_fc0, keep_prob=keep_prob, name="dropout")
            fc0 = tf.nn.relu(dropped, name="activation")
            end_points[end_point] = fc0
        
        # Logits
        inputs = end_points[end_point]
        in_dim = out_dim
        out_dim = num_classes
        end_point = 'Logits'
        with tf.variable_scope(end_point):
            W_fc1 = tf.get_variable("weights", [in_dim, out_dim], initializer=xavier)
            b_fc1 = tf.get_variable("biases", [out_dim], initializer=constant(0.0))
            with tf.name_scope('logits'):
                logits = tf.matmul(inputs, W_fc1) + b_fc1
            end_points[end_point] = logits

    return logits, end_points
