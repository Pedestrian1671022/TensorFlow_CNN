import tensorflow as tf
import tensorflow.contrib.slim as slim
from vgg_19 import vgg_19
from vgg_19 import vgg_arg_scope

checkpoint = 'vgg_19.ckpt'
train_tfrecord = "train.tfrecord"

image_pixels = 224
classes = 5
epochs = 100
train_size = 10935
batch_size = 50

def read_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"label":tf.FixedLenFeature([], tf.int64), "image":tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [image_pixels, image_pixels, 3])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features["label"], tf.int32)
    return img, label

images = tf.placeholder(tf.float32, [None, image_pixels, image_pixels, 3], name="input/x_input")
labels = tf.placeholder(tf.int64, [None], name="input/y_input")

with slim.arg_scope(vgg_arg_scope()):
    logits, end_points = vgg_19(images, num_classes=classes, is_training=True)

exclude = ["vgg_19/fc8"]
variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

one_hot_labels = slim.one_hot_encoding(labels, classes)
loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
total_loss = tf.losses.get_total_loss()
learning_rate = tf.Variable(initial_value=1e-4, trainable=False, name="learning_rate", dtype=tf.float32)
update_learning_rate = tf.assign(learning_rate, learning_rate*0.8)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss)
correct_prediction = tf.equal(labels, tf.argmax(end_points['vgg_19/fc8'], 1), name="correct_prediction")
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.train.Saver(variables_to_restore).restore(sess, checkpoint)
    # ckpt = tf.train.get_checkpoint_state("ckpt")
    # if ckpt:
    #     print(ckpt.model_checkpoint_path)
    #     tf.train.Saver(var_list=slim.get_variables_to_restore()).restore(sess, ckpt.model_checkpoint_path)
    # else:
    #     raise ValueError('The ckpt file is None.')
    tf.summary.FileWriter("logs/", sess.graph)
    dataset_train = tf.data.TFRecordDataset(train_tfrecord)
    dataset_train = dataset_train.map(read_and_decode)
    dataset_train = dataset_train.repeat(epochs).shuffle(1000).batch(batch_size)
    iterator_train = dataset_train.make_initializable_iterator()
    next_element_train = iterator_train.get_next()
    sess.run(iterator_train.initializer)

    for epoch in range(epochs):
        if epoch != 0 and epoch % 2 == 0:
            sess.run(update_learning_rate)
        print("learning_rate:", sess.run(learning_rate))
        for step in range(int(train_size/batch_size)):
            img_train, label_train = sess.run(next_element_train)
            _, _total_loss, _accuracy = sess.run([train_step, total_loss, accuracy], feed_dict={images:img_train, labels:label_train})
            if step % 10 == 0:
                print("step:", int(step / 10), "  total_loss:", _total_loss, " accuracy:", _accuracy)
        tf.train.Saver().save(sess, "ckpt/model.ckpt")
        print("Save ckpt:", epoch)