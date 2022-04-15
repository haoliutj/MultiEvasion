"""
fixed width and height, padding with 0 if needed
"""

import os, array
import numpy as np
import imageio
import argparse


def convert2image(filepath=None,width:int=320,height=320,padding_symbol=0):
    """
    convert binary file to image
    :param file:
    :return:
    """
    size = os.path.getsize(filepath)
    # residual = size % width # get the rest of content that the size is not enough for one row (padding could help)

    ## read binary file
    binary_file = open(filepath, 'rb')

    ## define binary array
    arr = array.array('B')
    arr.fromfile(binary_file, size)
    ## trim/pad
    if size >= width*height:
        arr = arr[:width*height]
    else:
        arr = list(arr) + [padding_symbol] * (width*height-size)
        arr = np.array(arr)
    binary_file.close()

    ## convert to image
    image = np.reshape(arr, (height, width))
    image = np.uint8(image)

    return image

def saveImage(image=None,path=None,filename=None):
   """

   :param image:
   :param path: parent path
   :return:
   """
   # if not os.path.exists(path):
   #     os.makedirs(path)
   os.makedirs(path,exist_ok=True)

   imageio.imwrite(path + filename, image)
   print(f'image saved in {path + filename}')


def main(inputpath, outputpath,width=320,height=320):
    items = os.listdir(inputpath)
    for item in items:

        ## convert to image
        image = convert2image(filepath=inputpath+item,width=width,height=height)
        ## save image
        saveImage(image=image,path=outputpath,filename=item + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='binary converts to image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_path', default='../data/all_file/', type=str, help='binary files folder path')
    parser.add_argument('--output_path', default='../data/all_image/', type=str, help='image folder path')
    args = parser.parse_args()  ## global variables
    print('\n', args, '\n')

    main(args.input_path,args.output_path)


