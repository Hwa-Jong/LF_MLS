import torch.utils.data as data
import torch
import numpy as np
import random
import csv
import os
import h5py
import utils


class CustomTrainDataset(data.Dataset):
    def __init__(self, file_path, scale, patch_size, an):
        super(CustomTrainDataset, self).__init__()
        
        self.HR, self.LR, self.names  = self.get_data(file_path, scale)
        
        self.scale = scale        
        self.psize = patch_size
        self.an = an
    
    def get_data(self, path, scale):
        f = open(os.path.join(path, '__names_x%d.csv'%scale),'r')
        rdr = csv.reader(f)
        names = []
        for line in rdr:
            names.append(line[0])        
        f.close()

        # load data hdf5
        f = h5py.File(os.path.join(path, 'Train_x%d.hdf5'%scale))
        hr = f.get('HR')
        lr = f.get('x%dLR'%scale)
        return hr, lr, names

    def get_targetNreferences_position(self, u, v):
        tar_pos = (u, v)
        refs_pos = []
        for u_ in range(u-1, u+2):
            for v_ in range(v-1, v+2):
                refs_pos.append((u_, v_))

        return tar_pos, refs_pos
    
    def __getitem__(self, index):
        
        s = index // (self.an * self.an) # set num
        name = self.names[s]
        idx = index % (self.an * self.an) # view index
        u = idx // self.an
        v = idx % self.an
                        
        # get one item
        tar_pos, refs_pos = self.get_targetNreferences_position(u, v)
        hr = self.HR[name]['data'][tar_pos]   
        H, W = hr.shape
        lrs = np.zeros((9, H//self.scale, W//self.scale))
        for i, ref_pos in enumerate(refs_pos):
            u_, v_ = ref_pos[0], ref_pos[1]
            if u_ < 0 or u_ >= self.an or v_ < 0 or v_ >= self.an:
                continue # include zero image
            
            lrs[i, :, :] = self.LR[name]['data'][u_, v_]
            

                                               
        # crop to patch
        H, W = hr.shape
        h, w = H//self.scale, W//self.scale
        lr_h = np.random.randint(0, h - self.psize + 1)
        lr_w = np.random.randint(0, w - self.psize + 1)
        hr_h = lr_h * self.scale
        hr_w = lr_w * self.scale

        hr = hr[hr_h:hr_h+(self.psize * self.scale), hr_w:hr_w+(self.psize * self.scale)]
        lrs = lrs[:, lr_h:lr_h+self.psize, lr_w:lr_w+self.psize]

        hr = hr[np.newaxis, ...] # ( c, H, W )
        lrs = lrs[:,np.newaxis, ...] #( refs_num, c, h, w )
        #normalize
        hr = utils.normalize(hr).astype(np.float32)
        lrs = utils.normalize(lrs).astype(np.float32)

        return hr, lrs

    def __len__(self):
        return len(self.names) * self.an * self.an


class CustomTestDataset(data.Dataset):
    def __init__(self, file_path, dataset_name, scale, an):
        super(CustomTestDataset, self).__init__()
        
        self.HR, self.LR, self.names  = self.get_data(file_path, dataset_name, scale)
        
        self.scale = scale        
        self.an = an
    
    def get_data(self, path, dataset_name, scale):
        n = '%s_x%d'%(dataset_name, scale)
        f = open(os.path.join(path, '%s__names.csv'%n),'r')
        rdr = csv.reader(f)
        names = []
        for line in rdr:
            names.append(line[0])        
        f.close()

        # load data hdf5
        f = h5py.File(os.path.join(path, '%s.hdf5'%n))
        hr = f.get('HR')
        lr = f.get('x%dLR'%scale)
        return hr, lr, names

    def get_targetNreferences_position(self, u, v):
        tar_pos = (u, v)
        refs_pos = []
        for u_ in range(u-1, u+2):
            for v_ in range(v-1, v+2):
                refs_pos.append((u_, v_))

        return tar_pos, refs_pos
    
    def __getitem__(self, index):
        
        s = index // (self.an * self.an) # set num
        name = self.names[s]
        idx = index % (self.an * self.an) # view index
        u = idx // self.an
        v = idx % self.an
                        
        # get one item
        tar_pos, refs_pos = self.get_targetNreferences_position(u, v)
        hr = self.HR[name]['data'][tar_pos]      

        H, W = hr.shape
        lrs = np.zeros((9, H//self.scale, W//self.scale))
        for i, ref_pos in enumerate(refs_pos):
            u_, v_ = ref_pos[0], ref_pos[1]
            if u_ < 0 or u_ >= self.an or v_ < 0 or v_ >= self.an:
                continue
            lrs[i, :, :] = self.LR[name]['data'][u_, v_]



        hr = hr[np.newaxis, ...] # ( 1, H, W )
        lrs = lrs[:,np.newaxis, ...] #( refs_num, 1, h, w )
        #normalize
        hr = utils.normalize(hr).astype(np.float32)
        lrs = utils.normalize(lrs).astype(np.float32)

        return hr, lrs

    def __len__(self):
        return len(self.names) * self.an * self.an