import os
import numpy as np
import h5py
import cv2
import utils
from PIL import Image
import csv



def main():
    test_dataset = 'HCI_new' # HCI_old HCI_new EPFL Stanford INRIA
    root_path = os.path.join('./dataset/Test', test_dataset)
    save_path = './dataset/Test_hdf5'

    scale = 2

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    names = os.listdir(root_path)
    names.sort()

    data_num = len(names)

    f = open(os.path.join(save_path, '%s_x%d__names.csv'%(test_dataset, scale)),'w', newline='')
    wr = csv.writer(f)
    for i, name in enumerate(names):
        wr.writerow([name])
    f.close()

    f = h5py.File(os.path.join(save_path, '%s_x%d.hdf5'%(test_dataset, scale)), 'w')
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

    #crop angle
    if scale == 4:
        crop_angle = 7
        start = (angle - crop_angle)//2
        data_hr = data_hr[start:start+crop_angle, start:start+crop_angle, ...]
        data_lr = data_lr[start:start+crop_angle, start:start+crop_angle, ...]
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

"""def get_lr(img, scale):
    H, W = img.shape[:2]
    h, w = H//scale, W//scale
    lr = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    return lr"""


if __name__ == "__main__":
    main()

print('finish')