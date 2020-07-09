import tensorflow as tf
from tensorflow.python.framework import tensor_shape

import numpy as np
from math import pi


# def random_shift_and_crop(image, roi_param, random_shift_delta):
#   '''Randomly shift the ROI by "random_shift_delta" pixels. Then crop the roi.
#      This is implemented as (1) extract a larger roi, roi_padded. 
#      (2) crop a random roi out from the roi_padded.'''
#   if random_shift_delta != 0:   # When need to SHIFT   
#     roi_param_padded = {}
#     roi_param_padded['roi_offset_x'] = roi_param['roi_offset_x'] - random_shift_delta
#     roi_param_padded['roi_offset_y'] = roi_param['roi_offset_y'] - random_shift_delta
#     roi_param_padded['roi_width'] = roi_param['roi_width'] + 2*random_shift_delta
#     roi_param_padded['roi_height'] = roi_param['roi_height'] + 2*random_shift_delta
#     roi_padded = extract_roi(image, roi_param_padded)
#     # Randomly crop a fixe-sized ROI within the "roi_padded"
#     random_crop_size = [roi_param['roi_height'], roi_param['roi_width'], 3]
#     roi = tf.image.random_crop(roi_padded, random_crop_size)
#   else:  # When NO need to SHIFT (the 'if' portion also handles the else case, but it requires a few extra steps)
#     roi = extract_roi(image, roi_param)
#   return roi

def augment_random_shift(image, image_width, image_height, max_x, max_y):
  '''Randomly shift the image by maximum of "max_x" and "max_y" pixels.'''
  if max_x == 0 and max_y == 0:
    return image
  else:   # When need to SHIFT   
    offset_height = max_y # Adds offset_height rows of zeros on top
    offset_width = max_x  # Adds offset_width columns of zeros on the left
    target_height = image_height + 2*max_y
    target_width = image_width + 2*max_x 
    image_padded = tf.image.pad_to_bounding_box(
      image, offset_height, offset_width, target_height, target_width
    )
    random_shift_x = tf.random_uniform(tensor_shape.scalar(), minval=0, maxval=2*max_x, dtype=tf.int32)
    random_shift_y = tf.random_uniform(tensor_shape.scalar(), minval=0, maxval=2*max_y, dtype=tf.int32)
    offset_height = random_shift_y
    offset_width = random_shift_x
    target_height = image_height
    target_width = image_width
    image_shifted = tf.image.crop_to_bounding_box(
        image_padded,
        offset_height,
        offset_width,
        target_height,
        target_width
    )
    return image_shifted

def extract_roi(image, roi_param):
  """
  This op cuts a rectangular part out of image. The top-left corner of the returned image is at offset_height, offset_width in image, and its lower-right corner is at offset_height + target_height, offset_width + target_width
  Input: 
     image: 4-D Tensor of shape [batch, height, width, channels] 
         or 3-D Tensor of shape [height, width, channels]"""

  # Handel the case when cropping is not needed --> skip cropping  
  if roi_param['roi_height'] == -1 and roi_param['roi_width'] == -1:
    print('No need to extract ROI, when roi_height and roi_width are -1!')
    return image  

  roi = tf.image.crop_to_bounding_box(
    image,
    roi_param['roi_offset_y'],
    roi_param['roi_offset_x'],
    roi_param['roi_height'],
    roi_param['roi_width']
  )
  return roi


def augment_random_rotate(image, random_rotate):
  # print('IMAGES RANK +++++++++++++', images.get_shape().ndims)
  if random_rotate != 0:
    # add rotate transformation to augment images
    rotate_degree = tf.random_uniform(tensor_shape.scalar(), minval=-random_rotate, maxval=random_rotate)
    rotate_radians = rotate_degree * pi / 180
    # THE FOLLOWING FUNCTION REQUIRES THE RANK OF THE INPUT "images" TO BE KNOWN. 
    image = tf.contrib.image.rotate(image, rotate_radians, interpolation='BILINEAR') 

  # if random_rotate:
  #   k = np.random.random_integers(0,3)
  #   images = tf.image.rot90(images, k)
  return image


def augment_flip(image, flip_left_right, flip_up_down):
  if flip_left_right:
    image = tf.image.random_flip_left_right(image)
  if flip_up_down:
    image = tf.image.random_flip_up_down(image)
  return image


def augment_brightness(image, random_brightness):
  # Adjust brightness using a delta randomly picked in the interval [-random_brightness, random_brightness)
  # random_brightness should be in the range [0,1)
  if random_brightness != 0:   
    image = tf.image.random_brightness(image, random_brightness)
  return image 


def input_image_resize(images, input_height, input_width):
  # input image resize to the network's expected input size
  # input image: 4-D with shape [batch, height, width, channels] 
  resized = tf.image.resize_bilinear(images, [input_height, input_width])
  return resized


def input_image_normalization(augmented_image, input_mean, input_std):
  # input normalization 
  offset_image = tf.subtract(augmented_image, input_mean)
  normalized = tf.multiply(offset_image, 1.0 / input_std)
  # normalized = tf.expand_dims(normalized, 0, name='DistortResult')
  return normalized


def zoom_central(image, minimum_central_fraction):

  central_fraction = random.uniform(minimum_central_fraction,1)

  if (minimum_central_fraction!=1 and minimum_central_fraction>0 and minimum_central_fraction<1):
    return tf.image.central_crop(image, central_fraction)
  return image


# def image_augmentation(jpeg_data, flip_left_right, flip_up_down, random_crop, random_scale,
#                           random_brightness, random_rotate, input_width, input_height,
#                           input_depth, roi_param, random_shift_delta, zoom_in):
#   """Creates the operations to apply the specified distortions.

#   During training it can help to improve the results if we run the images
#   through simple distortions like crops, scales, and flips. These reflect the
#   kind of variations we expect in the real world, and so can help train the
#   model to cope with natural data more effectively. Here we take the supplied
#   parameters and construct a network of operations to apply them to an image.

#   Cropping
#   ~~~~~~~~

#   Cropping is done by placing a bounding box at a random position in the full
#   image. The cropping parameter controls the size of that box relative to the
#   input image. If it's zero, then the box is the same size as the input and no
#   cropping is performed. If the value is 50%, then the crop box will be half the
#   width and height of the input. In a diagram it looks like this:

#   <       width         >
#   +---------------------+
#   |                     |
#   |   width - crop%     |
#   |    <      >         |
#   |    +------+         |
#   |    |      |         |
#   |    |      |         |
#   |    |      |         |
#   |    +------+         |
#   |                     |
#   |                     |
#   +---------------------+

#   Scaling
#   ~~~~~~~

#   Scaling is a lot like cropping, except that the bounding box is always
#   centered and its size varies randomly within the given range. For example if
#   the scale percentage is zero, then the bounding box is the same size as the
#   input and no scaling is applied. If it's 50%, then the bounding box will be in
#   a random range between half the width and height and full size.

#   Args:
#     flip_left_right: Boolean whether to randomly mirror images horizontally.
#     random_crop: Integer percentage setting the total margin used around the
#     crop box.
#     random_scale: Integer percentage of how much to vary the scale by.
#     random_brightness: Integer range to randomly multiply the pixel values by.
#     graph.
#     input_width: Horizontal size of expected input image to model.
#     input_height: Vertical size of expected input image to model.
#     input_depth: How many channels the expected input image should have.
#     input_mean: Pixel value that should be zero in the image for the graph.
#     input_std: How much to divide the pixel values by before recognition.

#   Returns:
#     The jpeg input layer and the distorted result tensor.
#   """

#   # jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
#   decoded_image = tf.image.decode_image(jpeg_data, channels=input_depth) # changed to use the more generic decode_image function. # decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
#   decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
#   decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
   
#   # "augment_random_rotate" requires images to have a known rank. We use a resizing function to force the tensor rank to be known.
#   original_input_shape = [1080, 1440]
#   # Input of resize_bilinear is 4-D with shape [batch, height, width, channels]
#   decoded_image_4d = tf.image.resize_nearest_neighbor(decoded_image_4d,
#                                               original_input_shape)
#   # print('precropped_image RANK +++++++++++++', precropped_image.get_shape().ndims)
#   preprocessed_image_3d = tf.squeeze(decoded_image_4d, squeeze_dims=[0])



#   augmented_image = augment_flip(preprocessed_image_3d, flip_left_right, flip_up_down)

#   augmented_image = augment_brightness(augmented_image, random_brightness)

#   augmented_image = augment_random_rotate(augmented_image, random_rotate)

#   augmented_image = random_shift_and_crop(augmented_image, roi_param, random_shift_delta)

#   augmented_image = zoom_central(augmented_image, zoom_in)

#   # convert tensor from rank 3 to rank 4
#   augmented_image = tf.expand_dims(augmented_image, 0)


#   return augmented_image


def aug(image):
  """Creates the operations to apply the specified distortions.

  During training it can help to improve the results if we run the images
  through simple distortions like crops, scales, and flips. These reflect the
  kind of variations we expect in the real world, and so can help train the
  model to cope with natural data more effectively. Here we take the supplied
  parameters and construct a network of operations to apply them to an image.

  Cropping
  ~~~~~~~~

  Cropping is done by placing a bounding box at a random position in the full
  image. The cropping parameter controls the size of that box relative to the
  input image. If it's zero, then the box is the same size as the input and no
  cropping is performed. If the value is 50%, then the crop box will be half the
  width and height of the input. In a diagram it looks like this:

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  Scaling
  ~~~~~~~

  Scaling is a lot like cropping, except that the bounding box is always
  centered and its size varies randomly within the given range. For example if
  the scale percentage is zero, then the bounding box is the same size as the
  input and no scaling is applied. If it's 50%, then the bounding box will be in
  a random range between half the width and height and full size.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.
    graph.
    input_width: Horizontal size of expected input image to model.
    input_height: Vertical size of expected input image to model.
    input_depth: How many channels the expected input image should have.
    input_mean: Pixel value that should be zero in the image for the graph.
    input_std: How much to divide the pixel values by before recognition.

  Returns:
    The jpeg input layer and the distorted result tensor.
  """


  # decoded_image = tf.image.decode_image(image, channels=input_depth) # changed to use the more generic decode_image function. # decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  # decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  # decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
   
  # # "augment_random_rotate" requires images to have a known rank. We use a resizing function to force the tensor rank to be known.
  # original_input_shape = [1080, 1440]
  # # Input of resize_bilinear is 4-D with shape [batch, height, width, channels]
  # decoded_image_4d = tf.image. resize_nearest_neighbor(decoded_image_4d,
  #                                             original_input_shape)
  # # print('precropped_image RANK +++++++++++++', precropped_image.get_shape().ndims)
  # preprocessed_image_3d = tf.squeeze(decoded_image_4d, squeeze_dims=[0])



  # augmented_image = augment_flip(preprocessed_image_3d, flip_left_right, flip_up_down)

  # augmented_image = augment_brightness(augmented_image, random_brightness)
  random_rotate = 5 #0.7
  augmented_image = augment_random_rotate(image, random_rotate)  
  
  image_width, image_height = 1440, 1080
  shift_max_x = 10
  shift_max_y = 10
  augmented_image = augment_random_shift(augmented_image, image_width, image_height, shift_max_x, shift_max_y)  
  # roi_param = {
  #   'roi_offset_x':950,
  #   'roi_offset_y':650,
  #   'roi_width':224,
  #   'roi_height':224
  # }
  # augmented_image = extract_roi(augmented_image, roi_param)
  
  # random_shift_delta = 50 #7
  # augmented_image = random_shift_and_crop(augmented_image, roi_param, random_shift_delta)

  # augmented_image = zoom_central(augmented_image, zoom_in)

  # # convert tensor from rank 3 to rank 4
  # augmented_image = tf.expand_dims(augmented_image, 0)


  return augmented_image

# augmented_image = image_augmentation(jpeg_data, flip_left_right, flip_up_down, random_crop, random_scale,
#                           random_brightness, random_rotate, input_width, input_height,
#                           input_depth, roi_param, random_shift_delta, zoom_in)  
