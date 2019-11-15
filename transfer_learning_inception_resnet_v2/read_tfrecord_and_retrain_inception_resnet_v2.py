import tensorflow as tf
import tensorflow.contrib.slim as slim
from inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2

checkpoint = "inception_resnet_v2.ckpt"
train_tfrecord = "train.tfrecord"

image_pixels = 299
classes = 5
epochs = 100
train_size = 3894
batch_size = 50

def read_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"label":tf.FixedLenFeature([], tf.int64),
                                                                     "image":tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [image_pixels, image_pixels, 3])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features["label"], tf.int32)
    return img, label

images = tf.placeholder(tf.float32, [None, image_pixels, image_pixels, 3], name="input/x_input")
labels = tf.placeholder(tf.int64, [None], name="input/y_input")

with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(images, num_classes=classes, is_training=True)

exclude = ["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]
variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

one_hot_labels = slim.one_hot_encoding(labels, classes)
loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
total_loss = tf.losses.get_total_loss()
tf.summary.scalar("total_loss", total_loss)
learning_rate = tf.Variable(initial_value=1e-4, trainable=False, name="learning_rate", dtype=tf.float32)
update_learning_rate = tf.assign(learning_rate, learning_rate*0.8)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss)
correct_prediction = tf.equal(labels, tf.argmax(end_points["Predictions"], 1), name="correct_prediction")
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.train.Saver(variables_to_restore).restore(sess, checkpoint)
    # ckpt = tf.train.get_checkpoint_state("ckpt")
    # if ckpt:
    #     print(ckpt.model_checkpoint_path)
    #     tf.train.Saver(var_list=slim.get_variables_to_restore()).restore(sess, ckpt.model_checkpoint_path)
    # else:
    #     raise ValueError("The ckpt file is None.")
    writer = tf.summary.FileWriter("logs/", sess.graph)
    dataset_train = tf.data.TFRecordDataset([train_tfrecord])
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
        tf.train.Saver().save(sess, "ckpt/model.ckpt")
        print("save ckpt:", epoch)
        print("=======================================================================================")