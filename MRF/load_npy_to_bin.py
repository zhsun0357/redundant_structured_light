import numpy as np
import glob
import os

npy_filenames = glob.glob('/data/sen/SLDenoising/real_corr_left_10/*/*.npy')

for idx in range(len(npy_filenames)):
    file_name = npy_filenames[idx][:-4]
    print(file_name)
    
    data_np = np.load(npy_filenames[idx])#.astype(np.int32)
    
    if not os.path.exists(file_name+".bin"):
        data_np.tofile(file_name+".bin")
