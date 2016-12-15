import tensorflow as tf
import csv
from PIL import Image
import scipy.misc
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn.cross_validation import train_test_split
from time import sleep
# from scipy.misc import imread, imsave
# import pandas as pd

flags = tf.app.flags
FLAGS = flags.FLAGS

# for i in tqdm(range(2000)):
#    sleep(0.1)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

filename = 'driver_imgs_list.csv'
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile, delimiter = ',')
    subjects = []
    labelnames = []
    filenames = []
    for row in csvreader:
        subjects.append(row[0])
        labelnames.append(row[1])
        filenames.append(row[2])
del(filenames[0])
del(subjects[0])
del(labelnames[0])

print(filenames[0])
print(labelnames[0])
print(subjects[0])

# randomize = np.arange(len(x))
# np.random.shuffle(randomize)
#
# filenames = filenames[randomize]
# subjects = subjects[randomize]
# labelnames = labelnames[randomize]

c = list(zip(filenames, subjects, labelnames))
random.shuffle(c)
filenames, subjects, labelnames = zip(*c)



fullnames = []
for ln, fn in zip(labelnames, filenames):
    fullnames.append('./imgs/train/'+ln+'/'+fn)

# trainingdata = []
#
# for path in tqdm(fullnames):
#     img = imread(path)
#     trainingdata.append(img)
#
#x_train = np.array(trainingdata)
y_train = []
labelvals = {'c0':0, 'c1':1,'c2':2,'c3':3,'c4':4,'c5':5,'c6':6,'c7':7,'c8':8,'c9':9}

for label in labelnames:
    y_train.append(labelvals[label])

temp = np.zeros((len(y_train), 10))
for i in range(len(y_train)):
    temp[i, y_train[i]] = 1

y_train = temp

#print(list(map(int, list(y_train[2, :]))))

filename_queue = tf.train.string_input_producer(fullnames, shuffle = False)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
my_img = tf.image.decode_jpeg(value)
init_op = tf.initialize_all_variables()
desFile = "./tt.tfrecords"
with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    writer = tf.python_io.TFRecordWriter(desFile)


    for i in tqdm(range(len(filenames))):
        image = my_img.eval()
        image = scipy.misc.imresize(image, (60, 45))
        #print(image.shape)
        #scipy.misc.imsave('./img.png', image)

        #x = raw_input()
        imageInString = image.tostring()
        # print(list(map(int, y_train[i, :].tolist())))
        # print(type(y_train[i, :].tolist()))
        example = tf.train.Example(features=tf.train.Features(feature={'image_raw':_bytes_feature(imageInString),
                                                                           'label':_int64_feature(list(map(int, list(y_train[i, :]))))}))
        writer.write(example.SerializeToString())
        #Image._show(Image.fromarray(np.asarray(image)))
    writer.close()

    coord.request_stop()
    coord.join(threads)












