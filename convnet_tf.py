import tensorflow as tf
import cv2
import numpy as np

img = cv2.imread('/home/nik-96/Pictures/wallpaper1.jpg', 1)[:, :, ::-1]

image = tf.placeholder(dtype='float32', shape=(1, )+img.shape)
print(img.shape)
conv1 = tf.keras.layers.Conv2D(inputs=image, filters=16, kernel_size=(3, 3))
relu1 = tf.nn.relu(conv1)
conv2 = tf.keras.layers.Conv2D(relu1, 32, (3, 3))
relu2 = tf.nn.relu(conv2)
conv3 = tf.keras.layers.Conv2D(relu2, 64, (3, 3))
relu3 = tf.nn.relu(conv3)
dense1 = tf.keras.layers.Conv2D(inputs=relu3, units=(256, ), activation=tf.nn.relu)
q_values_t = tf.layers.dense(256, 4)

with tf.Session() as sess:

    output = sess.run([q_values_t], feed_dict={image: img})
    print(output.shape)
