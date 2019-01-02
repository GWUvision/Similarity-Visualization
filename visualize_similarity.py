import numpy as np
import os
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2
# Recent versions may have a different TF slim models directory, such as:
# import tensorflow.models.research.slim.nets.resnet_v2 as resnet_v2

# Specify which dataset to use and which network to load ('faces' or 'hotels')
# TODO: Include landmarks
# You should have first downloaded and decompressed the pretrained networks from:
# https://www2.seas.gwu.edu/~astylianou/similarity-visualization/faces.tar.gz
# https://www2.seas.gwu.edu/~astylianou/similarity-visualization/hotels.tar.gz
# If you didn't download these into the main directory, you'll need to change the "pretrained_net" variable
which_dataset = 'hotels'
pretrained_net = os.path.join(which_dataset,which_dataset)

mean_im_path = which_dataset + '_meanIm.npy'
im1_path = which_dataset+'1.jpg'
im2_path = which_dataset+'2.jpg'

# Each batch will have two 256x256 RGB images
image_batch = tf.placeholder(tf.float32, shape=[2, 224, 224, 3])

# Load the model
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    _, layers = resnet_v2.resnet_v2_50(image_batch, num_classes=128, is_training=True)

# Load the pre-trained network.
# NOTE: Pre-trained networks were trained w/ L2 normalization on output features.
variables_to_restore = []
for var in slim.get_model_variables():
    excluded = False
    if 'momentum' in var.op.name.lower():
        excluded = True
    if not excluded:
        variables_to_restore.append(var)

# Start a session.
# If you need to specify a GPU, you can pass in a tf.ConfigProto() to tf.Session()
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

restore_fn = slim.assign_from_checkpoint_fn(pretrained_net,variables_to_restore)
restore_fn(sess)

# Specify the features that we'll be grabbing for our visualization
last_conv = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/postnorm/Relu:0")) # last convolutional layer
output_feature = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/pool5:0")) # pooled output feature

def preprocess_im(im_path,mean_im_path):
    bgr_img = cv2.imread(im_path)
    bgr_img = cv2.resize(bgr_img, (224,224))
    mean_img = np.load(mean_im_path)
    img = bgr_img - mean_img
    return img

imgs = [preprocess_im(im,mean_im_path) for im in [im1_path,im2_path]]

def compute_similarity(conv1,out1,conv2,out2):
    conv1_normed = conv1 / np.linalg.norm(out1) / conv1.shape[0]
    conv2_normed = conv2 / np.linalg.norm(out2) / conv2.shape[0]

    im_similarity = np.zeros((conv1_normed.shape[0],conv1_normed.shape[0]))
    for zz in range(conv1_normed.shape[0]):
        repPx = np.matlib.repmat(conv1_normed[zz,:],conv1_normed.shape[0],1)
        im_similarity[zz,:] = np.multiply(repPx,conv2_normed).sum(axis=1)

    return np.sum(im_similarity,axis=1), np.sum(im_similarity,axis=0)

cv,ff = sess.run([last_conv, output_feature], feed_dict={image_batch: imgs})
sim1, sim2 = compute_similarity(cv[0].reshape(-1,cv[0].shape[-1]),ff[0],cv[1].reshape(-1,cv[1].shape[-1]),ff[1])
