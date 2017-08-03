import tensorflow as tf
from tensorflow.contrib import slim

def vgg(inputs, num_classes, is_training=True, keep_prob=0.5, scope='KTH_VGG'):
    with tf.variable_scope(scope, 'KTH_VGG', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.flatten, slim.fully_connected, slim.dropout], 
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu,
                                # weights_regularizer=slim.l2_regularizer(0.0005),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer()):
                conv1 = slim.conv2d(inputs, 32, [3, 3], scope='Conv1')
                pool1 = slim.max_pool2d(conv1, [2, 2], scope='Pool1')
                conv2 = slim.conv2d(pool1, 64, [3, 3], scope='Conv2')
                pool2 = slim.max_pool2d(conv2, [2, 2], scope='Pool2')
                conv3 = slim.conv2d(pool2, 128, [3, 3], scope='Conv3')
                pool3 = slim.max_pool2d(conv3, [2, 2], scope='Pool3')
                conv4 = slim.conv2d(pool3, 256, [3, 3], scope='Conv4')
                pool4 = slim.max_pool2d(conv4, [2, 2], scope='Pool4')
                
                flatten5 = slim.flatten(pool4, scope='Flatten5')
                
                fc6 = slim.fully_connected(flatten5, 256, scope='FC6')
                dropout6 = slim.dropout(fc6, keep_prob=keep_prob, is_training=is_training, scope='Dropout6')
                
                logits = slim.fully_connected(dropout6, num_classes, activation_fn=None, scope='Logits')
                
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                
                return logits, end_points

# X = tf.placeholder(dtype=tf.float32, shape=[None, 56, 56, 1])
# logits, end_points = kth_vgg(X, 2350)
# print(logits)
# print("=" * 60)
# print(end_points)