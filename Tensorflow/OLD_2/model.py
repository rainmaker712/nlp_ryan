import tensorflow as tf
import tensorflow.contrib.eager as tfe
import csv

from functools import reduce

#tfe.enable_eager_execution()

embed_size = 10
window_size = 3
dict_size = 461
batch_size = 10
max_length = 20
filter_size = window_size*embed_size


def transform(id_map):
    raw_dir = '/home/dataset/raw_data/thugsta_db.tsv'
    train_tfrecord_dir = './corpus_set.tfrecord'

    dataset_writer = tf.python_io.TFRecordWriter(train_tfrecord_dir)

    with open(raw_dir, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=u'\t')
        corpus = []
        for i, row in enumerate(reader):
            if i == 0: continue
            for ians in range(1, 6):
                contents = [id_map[w] if w in id_map else id_map['UNK'] for w in row[0].split(' ') if not w == '']
                answer = [id_map[w] if w in id_map else id_map['UNK'] for w in row[ians].split(' ') if not w == '']
                if answer:
                    print(contents)
                    print(answer)
                    example = tf.train.Example()
                    example.features.feature['contents'].int64_list.value.extend(contents)
                    example.features.feature['answer'].int64_list.value.extend(answer)
                    example.features.feature['contents_length'].int64_list.value.append(len(contents))
                    example.features.feature['answer_length'].int64_list.value.append(len(answer))
                    dataset_writer.write(example.SerializeToString())
            print('{} written'.format(i))



def create_id_map():
    raw_dir = '/home/dataset/raw_data/thugsta_db.tsv'

    dicts = {'UNK': 0}
    with open(raw_dir, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=u'\t')
        for row in reader:
            words = reduce(lambda x, y: x+' '+y, row).split(' ')
            for w in words:
                if w == '': continue
                if w in dicts:
                    dicts[w] += 1
                else:
                    dicts[w] = 1

    word_rank = sorted(dicts, key=dicts.__getitem__, reverse=True)
    word_ids = enumerate(word_rank)
    id_map = dict(map(reversed, word_ids))
    return id_map, word_rank



def proto_input_fn():
    ids = tf.constant([[1, 2, 3, 4, 3, 2, 1], [1, 2, 3, 2, 3, 2, 3]])
    label_ids = tf.constant([[0, 0, 0, 1, 3, 1, 3, 2], [0, 0, 0, 2, 1, 3, 4, 4]])
    return {'ids': ids,
            'init_label': label_ids[:, :window_size],
            'label_length': int(label_ids.shape[1])-window_size
            }, label_ids[:, window_size:]


def parser(serialized_example):
    feature = {
            'contents': tf.VarLenFeature(tf.int64),
            'answer': tf.VarLenFeature(tf.int64)}


    parsed_feature = tf.parse_single_example(serialized_example, feature)

    contents = tf.cast(parsed_feature['contents'], tf.int32)
    answer = tf.cast(parsed_feature['answer'], tf.int32)
    return contents, answer


def input_fn():
    train_tfrecord_dir = './corpus_set.tfrecord'
    dataset = tf.data.TFRecordDataset(train_tfrecord_dir).map(parser)
    dataset = dataset.batch(batch_size)

    itr = dataset.make_one_shot_iterator()

    contents, answer = itr.get_next()

    output_shape = tf.constant([batch_size, max_length], tf.int64)
    contents = tf.sparse_to_dense(contents.indices, output_shape, contents.values)
    answer = tf.sparse_to_dense(answer.indices, output_shape, answer.values)

    return {'ids': contents,
            'init_label': tf.zeros(shape=[batch_size, window_size], dtype=tf.int32),
            'label_length': int(answer.shape[1])
            }, answer


def model_fn(mode, features, labels):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    embeddings = tf.Variable(tf.truncated_normal([dict_size, embed_size]), trainable=False)

    # embedding
    input_embeds = tf.nn.embedding_lookup(embeddings, features['ids'])

    input_flat = tf.layers.flatten(input_embeds)
    input_flat = tf.expand_dims(input_flat, -1)

    init_label = tf.nn.embedding_lookup(embeddings, features['init_label'])
    init_label = tf.layers.flatten(init_label)
    init_label = tf.expand_dims(init_label, -1)

    if not PREDICT:
        label_onehot = tf.one_hot(labels, depth=dict_size, dtype=tf.float32)

    # encoder
    encoder_conv = tf.layers.conv1d(
            inputs=input_flat,
            filters=2*embed_size,
            kernel_size=filter_size,
            strides=embed_size,
            padding='same')

    encoder_glu = encoder_conv[:, :, embed_size:]*tf.nn.sigmoid(encoder_conv[:, :, :embed_size])

    #decoder
    next_ids = []
    outs = []
    for l in range(features['label_length']):
        if l == 0: decoder_input = init_label

        decoder_conv = tf.layers.conv1d(
                inputs=init_label,
                filters=2*embed_size,
                kernel_size=filter_size,
                strides=embed_size)


        decoder_glu = decoder_conv[:, :, embed_size:]*tf.nn.sigmoid(decoder_conv[:, :, :embed_size])

        tiled_decoder_glu = tf.tile(decoder_glu, [1, int(encoder_glu.shape[1]), 1])


        dot_prod = tf.matmul(encoder_glu, decoder_glu, transpose_b=True)

        attention = tf.nn.softmax(dot_prod, axis=1)

        z_plus_e = encoder_glu + input_embeds

        tiled_attention = tf.tile(attention, [1, 1, embed_size])

        c = tf.reduce_sum(tiled_attention*z_plus_e, axis=1)
        decoder_glu = tf.reshape(decoder_glu, [-1, embed_size])

        logits = tf.layers.dense(c+decoder_glu, dict_size)

        out = tf.nn.softmax(logits)

        next_id = tf.argmax(out, axis=1)
        next_embeds = tf.nn.embedding_lookup(embeddings, next_id)
        next_embeds = tf.expand_dims(next_embeds, -1)
        decoder_input = tf.concat([decoder_input[:, embed_size:], next_embeds], axis=1)

        next_ids.append(next_id)
        outs.append(logits)

    outs = tf.stack(outs, axis=1)
    next_ids = tf.stack(next_ids, axis=1)

    if TRAIN:
        global_step = tf.train.get_global_step()
        loss = tf.losses.softmax_cross_entropy(label_onehot, outs)
        train_op = tf.train.GradientDescentOptimizer(1e-2).minimize(loss, global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                train_op=train_op,
                loss=loss)

    elif EVAL:
        loss = tf.losses.softmax_cross_entropy(label_onehot, outs)
        eval_metric_ops = {'acc': tf.metrics.accuracy(labels, next_ids)}
        estimator_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    elif PREDICT:
        estimator_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'prediction': next_ids})

    else:
        raise Exception('estiamtor spec is invalid')

    return estimator_spec



def main(_):
    id_map, word_map = create_id_map()

    fairseq = tf.estimator.Estimator(model_fn, model_dir='./checkpoint/model')
#    fairseq.train(input_fn, steps=10000)
#    fairseq.evaluate(input_fn, steps=10)

    pred = fairseq.predict(input_fn)
    for p in pred:
        words = map(lambda i: word_map[i], p['prediction'])
        answers = ' '.join(words)
        print(answers)





if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

