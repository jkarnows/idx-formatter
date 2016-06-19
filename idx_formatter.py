import numpy as np
import Image
import gzip
import os
import struct

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)


def _read8(bytestream):
  dt = np.dtype(np.uint8)
  return np.frombuffer(bytestream.read(1), dtype=dt)


def decode(imgf, output_file, n, imgs=False):
  """
  Given a binary file, convert information into a csv or a directory of images

  If imgs is true, output_file is a directory that will hold images 
      directory MUST exist before this
  If imgs is false, output_file will be a csv file
  """
  with gzip.open(imgf) as bytestream:

    if imgs:
      savedirectory = output_file
    else:
      o = open(output_file, "w")

    magic_num = _read32(bytestream)[0]
    num_images = _read32(bytestream)[0]
    num_rows = _read32(bytestream)[0]
    num_cols = _read32(bytestream)[0]

    images = []

    for i in range(n):
      image = []
      for j in range(num_rows * num_cols):
        image.append(_read8(bytestream)[0])
      images.append(image)

    if imgs:
      for j,image in enumerate(images):
        saveimage = np.array(image).reshape((num_rows,num_cols))
        result = Image.fromarray(saveimage.astype(np.uint8))
        result.save(output_file + str(j) + '.png')
    else:
      for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
      o.close()


def encode(input_dir, output_filename, ext='.png'):
  """
  Convert a directory of images into a gzipped IDX file, formatted similarly to
  the standard MNIST files found here: http://yann.lecun.com/exdb/mnist/

  INPUTS
    input_dir           path to folder
    output_filename     the filename for the fzipped IDX file 
    ext                 image filename extension for input
  """

  # Ensure the input directory has a trailing slash
  if input_dir[-1] != '/':
    input_dir += '/'

  fs = [input_dir + x for x in np.sort(os.listdir(input_dir)) if ext in x]
  num_imgs = len(fs)
  output_file = open(output_filename, "wb")

  # Write items in the header
  # MNIST uses 2051 as a magic number in the header, so following convention here
  output_file.write(struct.pack('>i', 2051))       # Magic Number for train/test images
  output_file.write(struct.pack('>i', num_imgs))   # Number of images
  
  # Load the first image to get dimensions
  im = np.asarray(Image.open(fs[0]).convert('L'), dtype=np.uint32)
  r,c = im.shape                 

  # Write the rest of the header
  output_file.write(struct.pack('>i', r))          # Number of rows in 1 image
  output_file.write(struct.pack('>i', c))          # Number of columns in 1 image

  # For each image, record the pixel values in the binary file
  for img in range(num_imgs):
    im = np.asarray(Image.open(fs[img]).convert('L'), dtype=np.uint32)
    for i in xrange(im.shape[0]):
      for j in xrange(im.shape[1]):
        output_file.write(struct.pack('>B', im[i,j]))

  # Close the file
  output_file.close()

  # Gzip the file (as this is used in encoding)
  f_in = open(output_filename)
  f_out = gzip.open(output_filename + '.gz', 'wb')
  f_out.writelines(f_in)
  f_out.close()
  f_in.close()
  os.remove(output_filename)