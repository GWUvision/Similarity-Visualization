from image_ops import *
from similarity_ops import *
import os
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
which_dataset = 'faces'
pretrained_net = os.path.join(which_dataset,which_dataset)

# The mean image for each dataset is included in the repository.
mean_im_path = which_dataset + '_meanIm.npy'

# For this demo, there are two example images for each of the datasets.
im1_path = which_dataset+'1.jpg'
im2_path = which_dataset+'2.jpg'

# Each batch will have two 256x256 RGB images
image_batch = tf.placeholder(tf.float32, shape=[2, 224, 224, 3])

# Load the model
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    _, layers = resnet_v2.resnet_v2_50(image_batch, num_classes=128, is_training=True)

# Specify which variables to restore when loading the pretrained network.
variables_to_restore = [var for var in slim.get_model_variables()]

# Start a session.
# If you need to specify a GPU, you can pass in a tf.ConfigProto() to tf.Session()
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# Load the pre-trained network.
# NOTE: Pre-trained networks were trained w/ L2 normalization on output features.
restore_fn = slim.assign_from_checkpoint_fn(pretrained_net,variables_to_restore)
restore_fn(sess)

# Grab the output of the last convolutional layer
last_conv = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/postnorm/Relu:0"))

# For this demo, load two example images from the same class.
imgs = [preprocess_im(im,mean_im_path) for im in [im1_path,im2_path]]

# Run the images through the network, get last conv features
cv = sess.run(last_conv, feed_dict={image_batch: imgs})

# Compute the spatial similarity maps (returns a heatmap that's the size of the last conv layer)
heatmap1, heatmap2 = compute_spatial_similarity(cv[0].reshape(-1,cv[0].shape[-1]),cv[1].reshape(-1,cv[1].shape[-1]))

# Combine the images with the (interpolated) similarity heatmaps.
im1_with_similarity = combine_image_and_heatmap(load_and_resize(im1_path),heatmap1)
im2_with_similarity = combine_image_and_heatmap(load_and_resize(im2_path),heatmap2)

# Merge the two images into a single image and save it out
combined_image = pil_bgr_to_rgb(combine_horz([im1_with_similarity,im2_with_similarity]))
combined_image.save(os.path.join('.',which_dataset+'_similarity.jpg'))
