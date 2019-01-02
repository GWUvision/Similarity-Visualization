# Visualizing Deep Similarity Networks
This repository contains code to generate the similarity visualizations for deep similarity, or embedding, networks described in https://TODO.com (WACV 2019).

<p align="center">
  <img width=50% src="https://www2.seas.gwu.edu/~astylianou/images/similarity-visualization/similarity.png">
</p>

<b>Abstract</b>: For convolutional neural network models that optimize an image embedding, we propose a method to highlight the regions of images that contribute most to pairwise similarity. This work is a corollary to the visualization tools developed for classification networks, but applicable to the problem domains better suited to similarity learning. The visualization shows how similarity networks that are fine-tuned learn to focus on different features. We also generalize our approach to embedding networks that use different pooling strategies and provide a simple mechanism to support image similarity searches on objects or sub-regions in the query image.

## Dependencies
This code was run using the python libraries and versions listed in requirements.txt.

To install these dependencies, run:

```
pip install -r requirements.txt
```

Other library versions may work, but have not been tested.

## Pretrained Models
This code comes with example images from the [Hotels-50k](https://github.com/GWUvision/Hotels-50K) and [VGG-Faces2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) datasets.

To generate the similarity visualizations, first download the pre-trained models:
* Hotels: https://www2.seas.gwu.edu/~astylianou/similarity-visualization/hotels.tar.gz
* Faces: https://www2.seas.gwu.edu/~astylianou/similarity-visualization/faces.tar.gz

## Code
The main function to generate the similarity visualizations can be found in similarity_ops.py. The function, ```compute_spatial_similarity``` takes in the outputs of the final convolutional layer from an embedding network for a pair of images and returns two spatial similarity maps, one that explains which parts of the first image make it look like the second image, and one that explains which parts of the second image make it look like the first image.

There are functions in image_ops.py that help interpolate and combine the original images with the similarity maps.

The code in visualize_similarity.py provides an end to end demonstration, using either the Hotels-50K or VGG-Faces2 pretrained networks, of how to extract the output from the last convolutional layer using TensorFlow and generate the spatial similarity maps.

## Citation
To cite this work, please use:

```
@inproceedings{stylianouSimVis2019,
  author = {Stylianou, Abby and Souvenir, Richard and Pless, Robert},
  title = {Visualizing Deep Similarity Networks},
  booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year = {2019}
}
```
