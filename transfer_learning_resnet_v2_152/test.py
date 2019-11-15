import tensorflow as tf
from tensorflow.contrib import slim
from resnet_utils import resnet_arg_scope
from resnet_v2_152 import resnet_v2_152

image_pixels = 224
classes = 5
validation_size = 4686
batch_size = 5
validation_tfrecord = "flowers_validation.tfrecord"

def read_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"label":tf.FixedLenFeature([], tf.int64), "image":tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [image_pixels, image_pixels, 3])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features["label"], tf.int32)
    return img, label

images = tf.placeholder(tf.float32, [None, image_pixels, image_pixels, 3], name="input/x_input")
labels = tf.placeholder(tf.int64, [None], name="input/y_input")

with slim.arg_scope(resnet_arg_scope()):
    logits, end_points = resnet_v2_152(images, num_classes=classes, is_training=False)

correct_prediction = tf.equal(labels, tf.argmax(end_points['predictions'], 1), name="correct_prediction")
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state("ckpt")
    if ckpt:
        print(ckpt.model_checkpoint_path)
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('The ckpt file is None.')
    dataset_validation = tf.data.TFRecordDataset([validation_tfrecord])
    dataset_validation = dataset_validation.map(read_and_decode)
    dataset_validation = dataset_validation.repeat(1).shuffle(1000).batch(batch_size)
    iterator_validation = dataset_validation.make_initializable_iterator()
    next_element_validation = iterator_validation.get_next()
    sess.run(iterator_validation.initializer)
    acc = 0
    for _ in range(int(validation_size/batch_size)):
        img_validation, label_validation = sess.run(next_element_validation)
        acc += sess.run(accuracy, feed_dict={images: img_validation, labels: label_validation})
    print("acc:", acc/(int(validation_size/batch_size)))