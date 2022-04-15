import os, array
import numpy as np
import imageio
import argparse




def get_width(filename):
    """
    get width of image based file size
    :param filename:
    :return:
    """
    ## get size in kb
    size = (os.path.getsize(filename)) / 1000
    ## get width
    if size < 10:
        width = 32
    elif size >= 10 and size < 30:
        width = 64
    elif size >= 30 and size < 60:
        width = 128
    elif size >= 60 and size < 100:
        width = 256
    elif size >= 100 and size < 200:
        width = 384
    elif size >= 200 and size < 500:
        width = 512
    elif size >= 500 and size < 1000:
        width = 768
    else:
        width = 1024

    return width


def convert2image(filepath=None,width:int=32):
    """
    convert binary file to image
    :param file:
    :return:
    """
    size = os.path.getsize(filepath)
    residual = size % width # get the rest of content that the size is not enough for one row (padding could help)

    ## read binary file
    binary_file = open(filepath, 'rb')

    ## define binary array
    ## load (size-residual) length of file, (remove the last residual length bytes)
    arr = array.array('B')
    arr.fromfile(binary_file, size - residual)
    binary_file.close()

    ## convert to image
    image = np.reshape(arr, (len(arr)//width, width))
    image = np.uint8(image)

    return image

def saveImage(image=None,path=None,filename=None):
   """

   :param image:
   :param path: parent path
   :return:
   """
   if not os.path.exists(path):
       os.makedirs(path)

   imageio.imwrite(path + filename, image)
   print(f'image saved in {path + filename}')


def main(inputpath, outputpath):
    items = os.listdir(inputpath)
    for item in items:
        ## get width
        width = get_width(inputpath+item)
        ## convert to image
        image = convert2image(filepath=inputpath+item,width=width)
        ## save image
        saveImage(image=image,path=outputpath,filename=item + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='binary converts to image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_path', default='../data/all_file/', type=str, help='binary files folder path')
    parser.add_argument('--output_path', default='../data/image/', type=str, help='image folder path')
    args = parser.parse_args()  ## global variables
    print('\n', args, '\n')

    main(args.input_path,args.output_path)


