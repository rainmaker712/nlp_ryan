"""
Seq2Seq chatbot apply to..
"""

from lib import helpers
import numpy as np
import tensorflow as tf

"""
Vocab
"""

x = [[5, 7, 8], [6, 3], [3], [1]] #input
xt, xlen = helpers.batch(x) #0으로 패딩

"""
Building model
"""

tf.reset_default_graph()
sess = tf.InteractiveSession()

""" Model input and output 
중요한 것은 vocab_size
Dynamic RNN 모델은 재학습 전에 batch사이즈와 Seq 길이가 다른 것을 받아들이지만,
Vocab_size가 달라지면 재학습이 필요.
"""

PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

