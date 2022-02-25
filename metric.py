import utils
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def calc_metric(hr, sr):
  psnr = []
  ssim = []
  for i in range(len(hr)):
    hr_, sr_ = hr[i,0], sr[i,0]
    psnr.append(compare_psnr(hr_, sr_))    
    ssim.append(compare_ssim(hr_, sr_))

  psnr = np.array(psnr)
  ssim = np.array(ssim)

  return psnr.mean(), ssim.mean()
# --------------------------------------------------------------------------------------

