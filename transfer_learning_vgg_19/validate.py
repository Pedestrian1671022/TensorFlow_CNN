import os
import shutil
from PIL import Image
import tensorflow as tf
from tensorflow.contrib import slim
from vgg_19 import vgg_arg_scope, vgg_19

image_pixels = 224
classes = 5
flowers_224x224 = "flowers_224x224"

data = "data"
error = "error"
daisy = "daisy"
dandelion = "dandelion"
roses = "roses"
sunflowers = "sunflowers"
tulips = "tulips"

images = tf.placeholder(tf.float32, [None, image_pixels, image_pixels, 3], name="input/x_input")

with slim.arg_scope(vgg_arg_scope()):
    logits, end_points = vgg_19(images, num_classes=classes, is_training=False)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state("ckpt")
    if ckpt:
        print(ckpt.model_checkpoint_path)
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('The ckpt file is None.')
    if os.path.exists(os.path.join(data, daisy)):
        shutil.rmtree(os.path.join(data, daisy))
    if not os.path.exists(os.path.join(data, daisy)):
        os.makedirs(os.path.join(data, daisy))
    if os.path.exists(os.path.join(data, error)):
        shutil.rmtree(os.path.join(data, error))
    if not os.path.exists(os.path.join(data, error)):
        os.makedirs(os.path.join(data, error))
    for file in os.listdir(os.path.join(flowers_224x224, daisy)):
        image = os.path.join(os.path.join(flowers_224x224, daisy), file)
        img = Image.open(image)
        img = tf.decode_raw(img.tobytes(), tf.uint8)
        img = tf.reshape(img, [image_pixels, image_pixels, 3])
        img = tf.expand_dims(img, 0)
        img = tf.cast(img, tf.float32)
        result = sess.run(tf.argmax(end_points['vgg_19/fc8'], 1), feed_dict={images: img.eval()})
        print(file, result)
        if result == [0]:
            shutil.move(os.path.join(os.path.join(flowers_224x224, daisy), file), os.path.join(data, daisy), file)
        else:
            shutil.move(os.path.join(os.path.join(flowers_224x224, daisy), file), os.path.join(data, error), file)