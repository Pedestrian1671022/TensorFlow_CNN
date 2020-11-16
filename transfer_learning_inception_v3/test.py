import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from inception_v3 import inception_v3_arg_scope, inception_v3

image_pixels = 299
classes = 5
batch_size = 1
validation_size = 1668
validation_tfrecord = "flowers_validation.tfrecord"

def read_and_decode(serialized_example):
    features = tf.compat.v1.parse_single_example(serialized_example, features={"label":tf.compat.v1.FixedLenFeature([], tf.compat.v1.int64),
                                                                     "filename": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
                                                                     "image":tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)})
    img = tf.compat.v1.decode_raw(features["image"], tf.compat.v1.uint8)
    img = tf.compat.v1.reshape(img, [image_pixels, image_pixels, 3])
    img = tf.compat.v1.cast(img, tf.compat.v1.float32)
    label = tf.compat.v1.cast(features["label"], tf.compat.v1.int32)
    filename = tf.compat.v1.cast(features["filename"], tf.compat.v1.string)
    return img, filename, label

images = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, image_pixels, image_pixels, 3], name="input/x_input")
labels = tf.compat.v1.placeholder(tf.compat.v1.int64, [None], name="input/y_input")

with slim.arg_scope(inception_v3_arg_scope()):
    logits, end_points = inception_v3(images, num_classes=classes, is_training=False)

correct_prediction = tf.compat.v1.equal(labels, tf.compat.v1.argmax(end_points["Predictions"], 1), name="correct_prediction")
accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.compat.v1.float32), name="accuracy")

with tf.compat.v1.Session() as sess:
    # ckpt = tf.compat.v1.train.get_checkpoint_state("ckpt")
    # if ckpt:
    #     print(ckpt.model_checkpoint_path)
    #     tf.compat.v1.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    # else:
    #     raise ValueError("The ckpt file is None.")
    dataset_validation = tf.compat.v1.data.TFRecordDataset([validation_tfrecord])
    dataset_validation = dataset_validation.map(read_and_decode)
    dataset_validation = dataset_validation.repeat(1).shuffle(1000).batch(batch_size)
    iterator_validation = dataset_validation.make_initializable_iterator()
    next_element_validation = iterator_validation.get_next()
    sess.run(iterator_validation.initializer)
    acc = 0
    for _ in range(int(validation_size/batch_size)):
        img_validation, filename, label_validation = sess.run(next_element_validation)
        print(filename, label_validation)
#         cv2.imshow("windows", cv2.imread(filename[0].decode()))
        # cv2.imshow("windows", np.squeeze(img_validation))
        # cv2.waitKey(0)
    #     acc += sess.run(accuracy, feed_dict={images: img_validation, labels: label_validation})
    # print("acc:", acc/(int(validation_size/batch_size)))
