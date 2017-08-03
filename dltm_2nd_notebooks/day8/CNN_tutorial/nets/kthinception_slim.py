import tensorflow as tf
from tensorflow.contrib import slim
from collections import OrderedDict

def inception(inputs, num_classes, final_end_point=None, scope=None, is_training=True, keep_prob=0.8):
    
    # tf.truncated_normal_initializer의 형태를 간소화
    trunc_normal = lambda stddev: tf.truncated_normal_initializer(mean=0.0, stddev=stddev)

    # tf.contrib.layers.xavier_initializer의 형태를 간소화
    xavier = tf.contrib.layers.xavier_initializer()

    # tf.contrib.layers.xavier_initializer_conv2d의 형태를 간소화
    xavier_conv = tf.contrib.layers.xavier_initializer_conv2d()

    # tf.constant_initializer의 형태를 간소화
    constant = lambda value: tf.constant_initializer(value=value)
    
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = OrderedDict()

    with tf.variable_scope(scope, 'kthInception', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            
            #######################
            ## First Module (FM) ##
            #######################
            # FM_conv_0_1x1:  1x1 conv2d
            end_point = 'FM_conv_0_1x1'
            net = slim.conv2d(inputs=inputs, kernel_size=(2, 2), stride = 2, num_outputs=64, padding='VALID',
                              weights_initializer=xavier_conv, scope=end_point)
            end_points[end_point] = net
            if end_point == final_end_point: return net, end_points
            
            # FM_conv1_3x3: 3x3 conv2d
            end_point = 'FM_conv_1_3x3'
            net = slim.conv2d(inputs=net, kernel_size=(3, 3), num_outputs=192,
                              weights_initializer=xavier_conv, scope=end_point)
            end_points[end_point] = net
            if end_point == final_end_point: return net, end_points
            
            # FM_maxpool_2_3x3: 3x3 max_pool2d
            end_point = 'FM_maxpool_2_3x3'
            net = slim.max_pool2d(inputs=net, kernel_size=(3, 3), stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_end_point: return net, end_points
            
            #################################
            ## First Inception Module (FI) ##
            #################################            
            end_point = 'FI'
            
            with tf.variable_scope(end_point):
                ## Branch 0
                # Conv_0_1x1
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs=net, kernel_size=(1, 1), num_outputs=64,
                                           scope='Conv_0_1x1')
                ## Branch 1
                # Conv_0_1x1
                # Conv_1_3x3
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs=net, kernel_size=(1, 1), num_outputs=96,
                                           weights_initializer=xavier_conv, 
                                           scope='Conv_0_1x1')
                    branch_1 = slim.conv2d(inputs=branch_1, kernel_size=(3, 3), num_outputs=128,
                                           scope='Conv_1_3x3')
                ## Branch 2
                # Conv_0_1x1
                # Conv_1_5x5
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs=net, kernel_size=(1, 1), num_outputs=16,
                                           weights_initializer=xavier_conv, 
                                           scope='Conv_0_1x1')
                    branch_2 = slim.conv2d(inputs=branch_2, kernel_size=(5, 5), num_outputs=32,
                                           scope='Conv_1_5x5')
                ## Branch 3
                # Avgpool_0_3x3
                # Conv_1_1x1
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(inputs=net, kernel_size=(3, 3),
                                               scope='Avgpool_0_3x3')
                    branch_3 = slim.conv2d(inputs=branch_3, kernel_size=(1, 1), num_outputs=32,
                                           weights_initializer=xavier_conv,
                                           scope='Conv_1_1x1')
                
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_end_point: return net, end_points
                
            ##################################
            ## Second Inception Module (SI) ##
            ##################################            
            end_point = 'SI'
            
            with tf.variable_scope(end_point):
                ## Branch 0
                # Conv_0_1x1
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs=net, kernel_size=(1, 1), num_outputs=128,
                                           scope='Conv_0_1x1')
                ## Branch 1
                # Conv_0_1x1
                # Conv_1_3x3
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs=net, kernel_size=(1, 1), num_outputs=128,
                                           weights_initializer=xavier_conv, 
                                           scope='Conv_0_1x1')
                    branch_1 = slim.conv2d(inputs=branch_1, kernel_size=(3, 3), num_outputs=192,
                                           scope='Conv_1_3x3')
                ## Branch 2
                # Conv_0_1x1
                # Conv_1_5x5
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs=net, kernel_size=(1, 1), num_outputs=32,
                                           weights_initializer=xavier_conv, 
                                           scope='Conv_0_1x1')
                    branch_2 = slim.conv2d(inputs=branch_2, kernel_size=(5, 5), num_outputs=96,
                                           scope='Conv_1_5x5')
                ## Branch 3
                # Avgpool_0_3x3
                # Conv_1_1x1
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(inputs=net, kernel_size=(3, 3),
                                               scope='Avgpool_0_3x3')
                    branch_3 = slim.conv2d(inputs=branch_3, kernel_size=(1, 1), num_outputs=64,
                                           weights_initializer=xavier_conv,
                                           scope='Conv_1_1x1')
                
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_end_point: return net, end_points

            ########################
            ## Second Module (SM) ##
            ########################
            # SM_Maxpool_0_3x3:  3x3 max_pool2d
            end_point = 'SM_Maxpool_0_3x3'
            net = slim.max_pool2d(inputs=net, kernel_size=(3, 3), stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_end_point: return net, end_points
            
            #################################
            ## Third Inception Module (TI) ##
            #################################            
            end_point = 'TI'
            
            with tf.variable_scope(end_point):
                ## Branch 0
                # Conv_0_1x1
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs=net, kernel_size=(1, 1), num_outputs=192,
                                           scope='Conv_0_1x1')
                ## Branch 1
                # Conv_0_1x1
                # Conv_1_3x3
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs=net, kernel_size=(1, 1), num_outputs=96,
                                           weights_initializer=xavier_conv, 
                                           scope='Conv_0_1x1')
                    branch_1 = slim.conv2d(inputs=branch_1, kernel_size=(3, 3), num_outputs=208,
                                           scope='Conv_1_3x3')
                ## Branch 2
                # Conv_0_1x1
                # Conv_1_5x5
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs=net, kernel_size=(1, 1), num_outputs=16,
                                           weights_initializer=xavier_conv, 
                                           scope='Conv_0_1x1')
                    branch_2 = slim.conv2d(inputs=branch_2, kernel_size=(5, 5), num_outputs=48,
                                           scope='Conv_1_5x5')
                ## Branch 3
                # Avgpool_0_3x3
                # Conv_1_1x1
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(inputs=net, kernel_size=(3, 3),
                                               scope='Avgpool_0_3x3')
                    branch_3 = slim.conv2d(inputs=branch_3, kernel_size=(1, 1), num_outputs=64,
                                           weights_initializer=xavier_conv,
                                           scope='Conv_1_1x1')
                
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_end_point: return net, end_points            
            
            #######################
            ## Third Module (TM) ##
            #######################
            # TM_Avgpool_0_7x7:  7x7 max_pool2d
            end_point = 'TM_Avgpool_0_14x14'
            net = slim.avg_pool2d(inputs=net, kernel_size=(7, 7), stride=7, scope=end_point)
            end_points[end_point] = net
            if end_point == final_end_point: return net, end_points
            
            end_point = 'Logits'
            net = slim.dropout(inputs=net, is_training=is_training, keep_prob=keep_prob, scope='Dropout')
            net = slim.conv2d(inputs=net, kernel_size=(1, 1), stride=1, num_outputs=num_classes, 
                              activation_fn=None, scope=end_point)
            shape = net.get_shape().as_list()
            net = tf.reshape(net, [-1, shape[1]*shape[2]*shape[3]])
            end_points[end_point] = net
            if end_point == final_end_point: return net, end_points 
            
            
            
#             ##########################
#             ## Fully-Connected (FC) ##
#             ##########################
#             # FC_flatten
#             end_point = 'FC_flatten'
#             net = slim.flatten(inputs=net, scope=end_point)
#             end_points[end_point] = net
#             if end_point == final_end_point: return net, end_points
            
#             # FC_1
#             end_point = 'FC_1'
#             net = slim.fully_connected(inputs=net, num_outputs=2350, scope=end_point)
#             end_points[end_point] = net
#             if end_point == final_end_point: return net, end_points
            
#             # FC_dropout_2
#             end_point = 'FC_dropout_2'
#             net = slim.dropout(inputs=net, is_training=is_training, keep_prob=keep_prob, scope=end_point)
#             end_points[end_point] = net
#             if end_point == final_end_point: return net, end_points
            
#             # FC_3
#             end_point = 'logits'
#             net = slim.fully_connected(inputs=net, num_outputs=num_classes, 
#                                        activation_fn=None, scope=end_point)
#             end_points[end_point] = net
#             if end_point == final_end_point: return net, end_points
            
            
    return net, end_points

# X = tf.placeholder(dtype=tf.float32, shape=[None, 56, 56, 1])
# net_re, end_points_re = kth_inception(inputs=X, num_classes=2350, scope='test')
# print(net_re)
# print("="*60)
# print(end_points_re)
