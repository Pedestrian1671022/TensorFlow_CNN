import tensorflow as tf
import tensorflow.contrib.slim as slim
from inception_v3 import inception_v3_arg_scope, inception_v3

checkpoint = "inception_v3.ckpt"
train_tfrecord = "flowers_train.tfrecord"

image_pixels = 299
classes = 5
epochs = 100
train_size = 3894
batch_size = 50

def read_and_decode(serialized_example):
    features = tf.compat.v1.parse_single_example(serialized_example, features={"label":tf.compat.v1.FixedLenFeature([], tf.compat.v1.int64),
                                                                     "filename":tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
                                                                     "image":tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)})
    img = tf.compat.v1.decode_raw(features["image"], tf.compat.v1.uint8)
    img = tf.compat.v1.reshape(img, [image_pixels, image_pixels, 3])
    img = tf.compat.v1.cast(img, tf.compat.v1.float32)
    label = tf.compat.v1.cast(features["label"], tf.compat.v1.int32)
    return img, label

images = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, image_pixels, image_pixels, 3], name="input/x_input")
labels = tf.compat.v1.placeholder(tf.compat.v1.int64, [None], name="input/y_input")

with slim.arg_scope(inception_v3_arg_scope()):
    logits, end_points = inception_v3(images, num_classes=classes, is_training=True)

exclude = ["InceptionV3/Logits", "InceptionV3/AuxLogits"]
variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

one_hot_labels = slim.one_hot_encoding(labels, classes)
loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
total_loss = tf.compat.v1.losses.get_total_loss()
tf.compat.v1.summary.scalar("total_loss", total_loss)
learning_rate = tf.compat.v1.Variable(initial_value=1e-4, trainable=False, name="learning_rate", dtype=tf.compat.v1.float32)
update_learning_rate = tf.compat.v1.assign(learning_rate, learning_rate*0.8)
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.compat.v1.control_dependencies(update_ops):
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss)
correct_prediction = tf.compat.v1.equal(labels, tf.compat.v1.argmax(end_points["Predictions"], 1), name="correct_prediction")
accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.compat.v1.float32), name="accuracy")
tf.compat.v1.summary.scalar("accuracy", accuracy)
merged_summary = tf.compat.v1.summary.merge_all()

with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run()
    tf.compat.v1.train.Saver(variables_to_restore).restore(sess, checkpoint)
    # ckpt = tf.compat.v1.train.get_checkpoint_state("ckpt")
    # if ckpt:
    #     print(ckpt.model_checkpoint_path)
    #     tf.compat.v1.train.Saver(var_list=slim.get_variables_to_restore()).restore(sess, ckpt.model_checkpoint_path)
    # else:
    #     raise ValueError("The ckpt file is None.")
    writer = tf.compat.v1.summary.FileWriter("logs/", sess.graph)
    dataset_train = tf.compat.v1.data.TFRecordDataset([train_tfrecord])
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
            _total_loss, _accuracy, summary, _ = sess.run([total_loss, accuracy, merged_summary, train_step],
                                                          feed_dict={images: img_train, labels: label_train})
            writer.add_summary(summary, step)
            if step % 5 == 0:
                print("step:", step / 5, " total_loss:", _total_loss, " accuracy:", _accuracy)
        tf.compat.v1.train.Saver().save(sess, "ckpt/model.ckpt")
        print("save ckpt:", epoch)
        print("=======================================================================================")
