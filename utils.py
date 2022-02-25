import cv2
import numpy as np
from math import ceil
import os
import torch
import skimage.color as color
from PIL import Image
import matplotlib.pyplot as plt


ERASE_LINE = '\x1b[2K' # erase line command

SPLIT_LINE = '\n#------------------------------------------------------------------------------------------------------------------------------------'

def bgr2y(im_bgr):
    im_rgb = im_bgr[:,:,(2,1,0)]
    ycbcr = color.rgb2ycbcr(im_rgb) # y range 16 ~ 235
    return ycbcr[:,:,0], None

def bgr2ycbcr(im_bgr):
    im_rgb = im_bgr[:,:,(2,1,0)]
    ycbcr = color.rgb2ycbcr(im_rgb) # y range 16 ~ 235
    return ycbcr[:,:,0], ycbcr[:,:,1:]

def ycbcr2bgr(im_ycbcr):
    rgb = color.ycbcr2rgb(im_ycbcr) 
    bgr = rgb[:,:,(2,1,0)]
    return bgr

def normalize(img):
    return img/255

def denormalize(img):
    return img*255

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calc_mean_corner_edge_inner(arr, an): 
    # arr shape = ( scenes, views_row, views_col ) or arr shape = ( views_row, views_col )    
    assert len(arr.shape) == 2 or len(arr.shape) == 3, ' arr shape is not correct. shape must be ( scenes, views_row, views_col ) or ( views_row, views_col ).  now shape : %s' %(str(arr.shape))
    
    if len(arr.shape) == 3:   
        arr = np.mean(arr, axis=0)


    inner = arr[1:an-1, 1:an-1]
    edge = np.array([ arr[0, 1:an-1],  arr[an-1, 1:an-1], arr[1:an-1, 0],  arr[1:an-1, an-1] ])
    corner = np.array( [ arr[0,0], arr[0,an-1], arr[an-1,0], arr[an-1,an-1] ] )

    return corner.mean(), edge.mean(), inner.mean()

    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def model_save(epoch, model, optimizer, save_path):
    # epoch : now epoch
    state = {
        'epoch': epoch + 1, 
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, save_path)

def model_load(load_path, model=None, optimizer=None, map_location=None):
    if map_location is None:
        checkpoint = torch.load(load_path)
    else:
        checkpoint = torch.load(load_path, map_location=map_location)

    if model is not None:
        model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'] # start epoch    

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_next_dir_number(path):
    files = list(os.walk(path))[0][1]
    if len(files) == 0:
        results_number = 0
    else:
        files.sort()
        fnum = None
        while len(files) > 0:
            fname = files.pop()
            fnum = fname.split('_')[0]
            if fnum.isdigit():                
                break
        
        if fnum is None or not fnum.isdigit():
            results_number = 0
        else:
            results_number = int(fnum)+1

    return results_number

def get_color_map(psnr_value, size=(512,512), angle=(9,9)):
    psnr_value = np.reshape(psnr_value, angle)
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    plt.jet()
    mesh = ax.pcolormesh(psnr_value)
    fig.colorbar(mesh)
    return plt

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class my_log():
    def __init__(self, path):
        self.__path = path
        
    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, path):
        self.__path = path

    def logging(self, context, log_only=False):
        if not os.path.exists(self.__path):
            f = open(self.__path, 'w')
        else:
            f = open(self.__path, 'a')

        if isinstance(context, list):
            for data in context:
                f.write(str(data)+'\n')
                if not log_only:
                    print(data)
        elif isinstance(context, str):
            f.write(context+'\n')
            if not log_only:
                print(context)
        
        f.close()

    def show(self, apply_split=False):
        if not os.path.exists(self.__path):
            print('not exist file!! - path : %s' %self.__path)
            return None
        else:
            f = open(self.__path, 'r')
            text = []
            while True:
                line = f.readline()
                if not line: break
                line = line.strip()
                if apply_split:
                    line = line.split()
                print(line)
                text.append(line)
            f.close()

            return text
