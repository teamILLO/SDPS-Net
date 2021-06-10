"""
bmp2mat
"""
import glob
import os
import numpy as np
from PIL import Image
import scipy.io as io
 
#src_dir = '123/'
save_dir = 'npy_data/'
#file_list = glob.glob(src_dir+'*.jpg')  # get name list of all .png files    
print('Start...')
# initrialize
 
img = Image.open('spec_normal.bmp')
# save to .npy
res = np.array(img, dtype='uint16')
print('Saving data...')
if not os.path.exists(save_dir):
	os.mkdir(save_dir)
np.save(save_dir+'origin.npy', res)
print('Done.')
 
numpy_file = np.load('npy_data/origin.npy')
 
io.savemat('Normal_gt.mat',{'data':numpy_file})