import cv2
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def load_and_resize(im_path):
    """
    Loads an image and resizes to 224x224.
    """
    bgr_img = cv2.imread(im_path)
    bgr_img = cv2.resize(bgr_img, (224,224))
    return bgr_img

def preprocess_im(im_path,mean_im_path):
    """
    Takes in an the path to an image and a numpy array of the mean image for the dataset.
    Resizes the image to 224x224, subtracts off the mean image.
    Everything stays in BGR.
    """
    bgr_img = load_and_resize(im_path)
    mean_img = np.load(mean_im_path)
    img = bgr_img - mean_img
    return img

def pil_bgr_to_rgb(img):
    b, g, r = img.split()
    return Image.merge("RGB", (r, g, b))

def combine_image_and_heatmap(img,heatmap):
    """
    Takes in a numpy array for an image and the similarity heatmap.
    Blends the two images together and returns a np array of the blended image.
    """
    cmap = plt.get_cmap('jet') # colormap for the heatmap
    heatmap = heatmap - np.min(heatmap)
    heatmap /= np.max(heatmap)
    heatmap = cmap(np.max(heatmap)-heatmap)
    if np.max(heatmap) < 255.:
        heatmap *= 255

    heatmap_img = cv2.resize(heatmap,(224,224))
    bg = Image.fromarray(img.astype('uint8')).convert('RGBA')
    fg = Image.fromarray(heatmap_img.astype('uint8')).convert('RGBA')
    outIm = np.array(Image.blend(bg,fg,alpha=0.5))
    return outIm

def combine_horz(images):
    """
    Combines two images into a single side-by-side PIL image object.
    """
    images = [Image.fromarray(img.astype('uint8')) for img in images]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im
