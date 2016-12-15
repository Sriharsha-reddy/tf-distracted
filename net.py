import tensorflow as tf
import tqdm
import os
import numpy as np
import re
import time
import datetime

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial Learning Rate.')
flags.DEFINE_integer('num_epochs', 2000, 'Nummber of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 128, "Number of images in batch")
flags.DEFINE_string('train_dir', os.getcwd(),"""Directory where to write event logs """)
flags.DEFINE_integer('max_steps', 1000, "Number of iterations in total")

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100000
MAX_STEPS = 10000

train_file = "tt.tfrecords"
validation_file = "yoda.tfrecords"

TOWER_NAME = 'tower'
NUM_CLASSES = 10


def read_and_decode(filename):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename)

    features = tf.parse_single_example(serialized_example, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([10], tf.int64)
    })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    depth_major = tf.reshape(image, [3, 45, 60])
    img = tf.transpose(depth_major, [1, 2, 0])

    img = tf.cast(img, tf.float32)

    img = tf.image.rgb_to_grayscale(img)

    distorted_image = tf.image.random_brightness(img, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    float_image = tf.image.per_image_whitening(distorted_image)



    label = features['label']
    #print(label)

    return float_image, label


def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  # if not num_epochs: num_epochs = None
  filename = os.path.join(FLAGS.train_dir,train_file if train else validation_file)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])


    image, label = read_and_decode(filename_queue)

    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)

    return images, sparse_labels



def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))



def _variable_on_cpu(name, shape, initializer):
    with tf.device("/cpu:0"):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype = dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):

    dtype = tf.float32
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images):

    ##CONVOLUTION LAYER 1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape = [3, 3, 1, 32], stddev=5e-2, wd = 0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        temp = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(temp, name = scope.name)
        _activation_summary(conv1)

    ##MAX POOLING LAYER 1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    ##NORMALIZATION LAYER 1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    ##CONVOLUTION LAYER 2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape = [3, 3, 32, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        temp = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(temp, name=scope.name)
        _activation_summary(conv2)

    ##MAX POOLING LAYER 2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    ##NORMALIZATION LAYER 2
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    ##CONVOLUTION LAYER 3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights', shape = [3, 3, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        temp = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(temp, name = scope.name)
        _activation_summary(conv3)

    ##NORMALIZATION LAYER 3
    norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

    ##MAX POOLING LAYER 3
    pool3 = tf.nn.max_pool(norm3, ksize = [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    ##LOCAL 4
    with tf.variable_scope('local4') as scope:

        reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local4)


    ##LOCAL 5
    with tf.variable_scope('local5') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local5 = tf.nn.relu(tf.matmul(local4, weights) + biases, name = scope.name)
        _activation_summary(local5)

    ##softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', shape=[192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local5, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def lossfn(logits, labels):

    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_average_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_average_op

def train(total_loss, global_step):

    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)

    tf.scalar_summary('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads,global_step = global_step)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for var, grad in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name, grad)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def training():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images, labels = inputs(train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
        # images, labels = mnist.train.next_batch(FLAGS.batch_size)
        # images = tf.reshape(images, [-1, 28, 28, 1])


        #print(labels)
        #print(images)

        logits = inference(images)

        #print(logits)

        loss_val = lossfn(logits, labels)

        train_op = train(loss_val, global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)


        #np.savetxt('harsha.txt',logits, delimiter=',', newline='\n')

        for step in range(MAX_STEPS):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss_val])
            duration = time.time() - start_time

            assert not np.isnan(loss_value)

            if step%10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step/duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')

                print(format_str % (datetime.datetime.now(), step, loss_value,examples_per_sec, sec_per_batch))

                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                    if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)








def main(argv=None):  # pylint: disable=unused-argument
  training()


if __name__ == '__main__':
    tf.app.run()





#tensorboard --logdir=PycharmProjects/untitled









































