import os
import shutil
from PIL import Image
import tensorflow as tf
from tensorflow.contrib import slim
from inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2


image_pixels = 299
classes = 5
flowers_299x299 = "flowers_299x299"

data = "data"
error = "error"
daisy = "daisy"
dandelion = "dandelion"
roses = "roses"
sunflowers = "sunflowers"
tulips = "tulips"

images = tf.placeholder(tf.float32, [None, image_pixels, image_pixels, 3], name="input/x_input")

with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(images, num_classes=classes, is_training=False)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state("ckpt")
    if ckpt:
        print(ckpt.model_checkpoint_path)
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError("The ckpt file is None.")
    if os.path.exists(os.path.join(data, tulips)):
        shutil.rmtree(os.path.join(data, tulips))
    if not os.path.exists(os.path.join(data, tulips)):
        os.makedirs(os.path.join(data, tulips))
    if os.path.exists(os.path.join(data, error)):
        shutil.rmtree(os.path.join(data, error))
    if not os.path.exists(os.path.join(data, error)):
        os.makedirs(os.path.join(data, error))
    for file in os.listdir(os.path.join(flowers_299x299, tulips)):
        image = os.path.join(os.path.join(flowers_299x299, tulips), file)
        img = Image.open(image)
        img = tf.decode_raw(img.tobytes(), tf.uint8)
        img = tf.reshape(img, [image_pixels, image_pixels, 3])
        img = tf.expand_dims(img, 0)
        img = tf.cast(img, tf.float32)
        result = sess.run(tf.argmax(end_points["Predictions"], 1), feed_dict={images: img.eval()})
        print(file, result)
        if result == [4]:
            shutil.move(os.path.join(os.path.join(flowers_299x299, tulips), file), os.path.join(data, tulips), file)
        else:
            shutil.move(os.path.join(os.path.join(flowers_299x299, tulips), file), os.path.join(data, error), file)