#-*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_integer('batchSize', 100, 'batch size') # 배치 크기
tf.app.flags.DEFINE_integer('trainSteps', 20000, 'train steps') # 학습 에포크
tf.app.flags.DEFINE_float('dropoutWidth', 0.5, 'dropout width') # 드롭아웃 크기
tf.app.flags.DEFINE_integer('layerSize', 3, 'layer size') # 멀티 레이어 크기 (multi rnn)
tf.app.flags.DEFINE_integer('hiddenSize', 128, 'weights size') # 가중치 크기
tf.app.flags.DEFINE_float('learningRate', 1e-3, 'learning rate') # 학습률
tf.app.flags.DEFINE_string('dataPath', './data_in/ChatBotData.csv', 'data path') #  데이터 위치
tf.app.flags.DEFINE_string('vocabularyPath', './data_in/vocabularyData.voc', 'vocabulary path') # 사전 위치
tf.app.flags.DEFINE_string('checkPointPath', './data_out/checkpoint', 'check point path') # 체크 포인트 위치
tf.app.flags.DEFINE_integer('shuffleSeek', 1000, 'shuffle random seek') # 셔플 시드값
tf.app.flags.DEFINE_integer('maxSequenceLength', 25, 'max sequence length') # 시퀀스 길이
tf.app.flags.DEFINE_integer('embeddingSize', 128, 'embedding size') # 임베딩 크기
tf.app.flags.DEFINE_boolean('tokenizeAsMorph', True, 'set morph tokenize') # 형태소에 따른 토크나이징 사용 유무
tf.app.flags.DEFINE_boolean('embedding', True, 'Use Embedding flag') # 임베딩 유무 설정
tf.app.flags.DEFINE_boolean('multilayer', True, 'Use Multi RNN Cell') # 멀티 RNN 유무
# Define FLAGS
DEFINES = tf.app.flags.FLAGS
