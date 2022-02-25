import os
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_model_summary as tsummary
import numpy as np
from PIL import Image
import datetime
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.autograd.profiler as profiler

from models import LF_MLS
import models, dataset, utils, metric



def train(dataset_dir, result_dir, load_path, epochs, save_term, scale, device):
    # make results dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    results_number = utils.get_next_dir_number(result_dir)

    ### train setting
    model_type = 'LF_MLS'
    batch_size = 32
    crop_size = 64
    input_imgs_num = 9
    apply_scheduler = True
    lr_scheduler_step = [ i for i in range(50, 300, 50) ]
    lr_scheduler_gamma = 0.5
    lr = 1E-3
    center = input_imgs_num//2
    epsilon = 1e-8
    visual_num = 40
    lr_scheduler_last_epoch = -1
    start_epoch = 1
    an = 9

    #set dir name
    desc = 'LFSR'
    desc += '_'+model_type
    desc +='_x%d'%scale
    desc +='_%depoc'%epochs

    save_dir = os.path.join(result_dir, '%04d_' %(results_number) + desc)
    os.mkdir(save_dir)

    #ckpt dir
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    os.mkdir(ckpt_dir)

    # set my logger
    log = utils.my_log(os.path.join(save_dir, 'results.txt'))
    log.logging('< Info >')
    log.logging('img type is YCbCr. Cb and Cr are UPsampled by bicubic interpolation and Y is UPsampled by this network')

    # load data
    print('loading data...')
    train_set = dataset.CustomTrainDataset(file_path=os.path.join(dataset_dir, 'Train_hdf5'), scale=scale, patch_size=crop_size, an=9)
    train_generator = DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, shuffle=True)

    test_name = ['HCI_new', 'INRIA_Lytro'] # HCI_old HCI_new EPFL Stanford INRIA_Lytro

    valid_gen_list = []
    for name in test_name:
        print('load evaluate data : ', name)
        valid_set = dataset.CustomTestDataset(file_path=os.path.join(dataset_dir, 'Test_hdf5'), dataset_name=name, scale=scale, an=an)
        valid_generator = DataLoader(dataset=valid_set, batch_size=1, shuffle=False)
        valid_gen_list.append(valid_generator)

    model = LF_MLS(scale=scale) 
    model = model.to(device=device)

    summary_shape = (batch_size,input_imgs_num,1,64,64)
    log.logging(tsummary.summary(model, torch.zeros(summary_shape, dtype=torch.float32).to(device=device), batch_size=batch_size, show_input=False))
    
    log.logging(utils.SPLIT_LINE)
    # model details
    #log.logging('< model >')
    #log.logging('%s' %model, log_only=True)
    #log.logging(utils.SPLIT_LINE)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # lr scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_scheduler_step, gamma=lr_scheduler_gamma, last_epoch=lr_scheduler_last_epoch, verbose=True)

    # Info
    log.logging('Model Type : ' + str(model_type))
    log.logging('dataset path : ' + dataset_dir)
    log.logging('results path : '+ save_dir)
    log.logging('load model path : '+ str(load_path))
    log.logging('sr scale : %d' %(scale))
    log.logging('total epochs : %d' %epochs)
    log.logging('batch size  : %d' %batch_size)
    log.logging('optimizer : Adam(lr:%f)' %(lr))
    log.logging('optimizer scheduler : ' + str(apply_scheduler) + ' - step:%s, gamma:%.2f, last_epoch:%d' %(str(lr_scheduler_step), lr_scheduler_gamma, lr_scheduler_last_epoch))
    
    #loss
    criterion = nn.L1Loss()

    log.logging(utils.SPLIT_LINE, log_only=True)

    # model load
    print('model load...')
    if load_path is not None:
        model, optimizer, start_epoch = utils.model_load(load_path, model, optimizer, map_location=device)
        
    best_psnr = 0.0
    best_ssim = 0.0
    best_psnr_epoch = -1

    # train start
    train_start_time = time.time()
    log.logging('< train >')
    
    total_batch = len(train_generator)
    
    for epoch in range(start_epoch, epochs+1):      
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        epoch_start_time = time.time()
        model.train()

        for batch_idx, batch in enumerate(train_generator):
            hr, lrs = batch[0].to(device, dtype=torch.float32), batch[1].to(device, dtype=torch.float32)
            
            optimizer.zero_grad()   
            outputs = model(lrs)

            batch_loss = criterion(outputs, hr)
                
            batch_loss.backward()
            optimizer.step()
            
            batch_psnr, batch_ssim = metric.calc_metric(hr.detach().cpu().numpy(), outputs.detach().cpu().numpy())

            epoch_loss += batch_loss.detach().item()
            epoch_psnr += batch_psnr
            epoch_ssim += batch_ssim

            if batch_idx % max((10//batch_size),1) == 0:
                print('\r%d/%d << batch loss: %.5f || batch psnr: %.5f || batch ssim: %.5f >> time: %.1f sec' %(batch_idx, total_batch, epoch_loss/(batch_idx+1), epoch_psnr/(batch_idx+1), epoch_ssim/(batch_idx+1), time.time()-epoch_start_time), end='')

        print(utils.ERASE_LINE, end='')

        epoch_loss = epoch_loss / total_batch
        epoch_psnr = epoch_psnr / total_batch
        epoch_ssim = epoch_ssim / total_batch
        
        log.logging('\r[%03d/%03d] epoch << train loss: %.5f | psnr:%.3f | ssim:%.4f >> time: %.1f sec' %(epoch, epochs, epoch_loss, epoch_psnr, epoch_ssim, time.time()-epoch_start_time))        

        # evaluate
        test_start = time.time()
        model.eval()
        now_psnr = 0.0
        now_ssim = 0.0
        test_psnr = []
        test_ssim = []

        for idx in range(len(valid_gen_list)):
            test_gen = valid_gen_list[idx]
            test_batch = len(test_gen)
            scene_num = test_batch // (an * an)
            test_imgs_psnr = np.zeros((scene_num, an,an))
            test_imgs_ssim = np.zeros((scene_num, an,an))
            
            for batch_idx, batch in enumerate(test_gen):
                print('\r [%d/%d] %d%% calculate psnr&ssin ( test dataset %s )'%(idx+1, len(valid_gen_list), batch_idx / test_batch*100 , test_name[idx]), end='')

                hr, lrs = batch[0].to(device, dtype=torch.float32), batch[1].to(device, dtype=torch.float32)

                with torch.no_grad():
                    outputs = model(lrs)

                test_img_psnr, test_img_ssim = metric.calc_metric(hr.detach().cpu().numpy(), outputs.detach().cpu().numpy())
                scene, row, col = batch_idx//(an*an), (batch_idx//an)%an, batch_idx%an
                test_imgs_psnr[scene, row, col] = test_img_psnr
                test_imgs_ssim[scene, row, col] = test_img_ssim

            now_psnr += test_imgs_psnr.mean()
            now_ssim += test_imgs_ssim.mean()
            
            test_psnr.append(test_imgs_psnr)
            test_ssim.append(test_imgs_ssim)

        now_psnr = now_psnr/len(valid_gen_list)
        now_ssim = now_ssim/len(valid_gen_list)
        print('')
        
        # best model save
        if now_psnr > best_psnr:            
            best_psnr = now_psnr
            best_ssim = now_ssim
            best_psnr_epoch = epoch
            utils.model_save(epoch, model, optimizer, os.path.join(save_dir, 'best_psnr_model.pt'))
            log.logging('---------- best psnr model saved!! epoch:%d < best psnr:ssim = %.3f:%.4f > ----------' %(best_psnr_epoch, best_psnr, best_ssim))
            
        # show detail      
        for idx in range(len(valid_gen_list)):
            psnr_corner, psnr_edge, psnr_inner = utils.calc_mean_corner_edge_inner(test_psnr[idx], an)
            check_psnr_mean = (psnr_corner*4 + psnr_edge*28 + psnr_inner*49)/81
            ssim_corner, ssim_edge, ssim_inner = utils.calc_mean_corner_edge_inner(test_ssim[idx], an)
            check_ssim_mean = (ssim_corner*4 + ssim_edge*28 + ssim_inner*49)/81
            log.logging('\t < Detail >  %s || psnr/ssim: %.2f/%.3f ( corner:%.2f/%.3f edge:%.2f/%.3f inner:%.2f/%.3f mean:%.2f/%.3f )' %(test_name[idx], test_psnr[idx].mean(), test_ssim[idx].mean(), psnr_corner.round(2), ssim_corner.round(3), psnr_edge.round(2), ssim_edge.round(3), psnr_inner.round(2), ssim_inner.round(3), check_psnr_mean.round(2), check_ssim_mean.round(3)))
        
        log.logging('[%03d/%03d] epoch << test  psnr:%.2f ssim:%.3f >> time: %.1f sec' %(epoch, epochs, now_psnr.round(2), now_ssim.round(3), time.time()-test_start))        
        print('')
        

        # ckpt save
        if epoch % save_term == 0:      
            # ckpt model save
            save_name = os.path.join(ckpt_dir, 'model-%03d.pt' %(epoch))
            utils.model_save(epoch, model, optimizer, save_name)

        # lr scheduler
        if apply_scheduler:
            scheduler.step() 

    #train end
    if best_psnr_epoch is not None:
        log.logging('Best psnr')
        log.logging('epoch : %d' %best_psnr_epoch)
    log.logging('val psnr : %f' %best_psnr)

    log.logging('total train time : %s' %(datetime.timedelta(seconds=(time.time()-train_start_time))))

    print('train finished!')


    opt = {
        'batch_size':32,
        'lr':0.001,
        'apply_lr_scheduler':True,
        'lr_scheduler_step':[ i for i in range(50, 300, 50) ],
        'lr_scheduler_gamma':0.5,
        'crop_size':64,
        'refs_num':8,

        'model_type':'LF_MLS_noFM',
        'batch_norm':False,

        'loss_type':'L1',
        'custom_loss_alpha':0.0,
        'custom_loss_beta':0.0
    }
##################################################################################################################################################
################################################################### test #########################################################################
##################################################################################################################################################
def test(dataset_dir, dataset_name, result_dir, model_path, scale, device):
    results_number = model_path[:4]
    dir_name = os.path.join(result_dir, (results_number+'_LFSR_Evaluate_x%d'%scale))

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    ### test setting
    an = 9

    print('load evaluate data : ', dataset_name)

    valid_set = dataset.CustomTestDataset(file_path=os.path.join(dataset_dir, 'Test_hdf5'), dataset_name=dataset_name, scale=scale, an=an)
    valid_generator = DataLoader(dataset=valid_set, num_workers=1, batch_size=1, shuffle=False)

    model = LF_MLS(scale=scale)

    model, _, _ = utils.model_load(os.path.join(result_dir, model_path), model, map_location=device)
    model = model.to(device=device)

    log = utils.my_log(os.path.join(dir_name, '%s_evaluate.txt'%dataset_name))
    log.logging('< %s >' %(datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")))        

    # evaluate
    eval_start = time.time()
    model.eval()

    valid_batch = len(valid_generator)
    valid_imgs_psnr = np.zeros((valid_batch//(an*an), an, an))
    valid_imgs_ssim = np.zeros((valid_batch//(an*an), an, an))

    for batch_idx, batch in enumerate(valid_generator):        
        print('\r  %d%% calculate psnr&ssin ( validation dataset %s )'%(batch_idx / valid_batch*100 , dataset_name), end='')
        hr, lrs = batch[0].to(device, dtype=torch.float32), batch[1].to(device, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(lrs)
        valid_img_psnr, valid_img_ssim = metric.calc_metric(hr.detach().cpu().numpy(), outputs.detach().cpu().numpy())

        scene, row, col = batch_idx//(an*an), (batch_idx//an)%an, batch_idx%an
        valid_imgs_psnr[scene, row, col] = valid_img_psnr
        valid_imgs_ssim[scene, row, col] = valid_img_ssim

    
    for_colormap = valid_imgs_psnr.mean(0)
    plt = utils.get_color_map(for_colormap, angle=(an,an))
    plt.savefig(os.path.join(dir_name, '%s_colormap.png'%dataset_name))
    print('')
    
    psnr_corner, psnr_edge, psnr_inner = utils.calc_mean_corner_edge_inner(valid_imgs_psnr, an)
    ssim_corner, ssim_edge, ssim_inner = utils.calc_mean_corner_edge_inner(valid_imgs_ssim, an)

    log.logging('%s || psnr_mean: %.5f ssim_mean: %.5f || corner:%.5f/%.5f edge:%.5f/%.5f inner:%.5f/%.5f || time: %.1f sec' %(dataset_name, valid_imgs_psnr.mean(), valid_imgs_ssim.mean(), psnr_corner,ssim_corner, psnr_edge,ssim_edge, psnr_inner,ssim_inner, time.time()-eval_start))        

    print('test finished!')

# -----------------------------------------------------------------------------------------------------------------------------------------------------
