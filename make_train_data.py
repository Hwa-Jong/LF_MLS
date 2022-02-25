import os
import numpy as np
import h5py
import cv2
import utils
from PIL import Image
import csv



def main():
    root_path = './dataset/Train'
    save_path = './dataset/Train_hdf5'

    scale = 2

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    names = os.listdir(root_path)
    names.sort()

    data_num = len(names)

    f = open(os.path.join(save_path, '__names_x%d.csv'%scale),'w', newline='')
    wr = csv.writer(f)
    for i, name in enumerate(names):
        wr.writerow([name])
    f.close()

    f = h5py.File(os.path.join(save_path, 'Train_x%d.hdf5'%(scale)), 'w')
    f.create_group('HR')
    f.create_group('x%dLR'%scale)

    for i, name in enumerate(names):
        data_hr, data_lr = get_data(os.path.join(root_path, name), scale)
        
        # save hdf5
        hr_group = f['HR'].create_group(name)
        hr_group.create_dataset('data', data=data_hr)

        lr_group = f['x%dLR'%scale].create_group(name)
        lr_group.create_dataset('data', data=data_lr)

        print('\r%d/%d ...' %(i+1, data_num), end='')

    f.close()
    print('\nend')



def get_data(path, scale):
    angle = 9

    data_hr = []
    data_lr = []
    for u in range(angle):
        tmp_hr = []
        tmp_lr = []
        for v in range(angle):
            hr = cv2.imread(os.path.join(path, '%d.png'%(u*9+v)))
            hr = preprocessing(hr, scale)
            lr = get_lr(hr, scale)
            tmp_hr.append(hr)
            tmp_lr.append(lr)
        data_hr.append(tmp_hr)
        data_lr.append(tmp_lr)

    data_hr = np.array(data_hr)
    data_lr = np.array(data_lr)

    return data_hr, data_lr
        

def preprocessing(img, scale):
    H, W = img.shape[:2]

    #bgr2y
    img, _ = utils.bgr2y(img)

    #cutout
    H = H//scale*scale
    W = W//scale*scale
    img = img[:H, :W]
    
    return img


# downsampling using PIL.Image
def get_lr(img, scale):
    hr = Image.fromarray(img)
    W, H = hr.size
    w, h = W//scale, H//scale
    lr = hr.resize((w, h), Image.BICUBIC)
    lr = np.array(lr)
    return lr

if __name__ == "__main__":
    main()