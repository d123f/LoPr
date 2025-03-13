import numpy as np
import os
import random
import glob
import pandas as pd 
import h5py
import SimpleITK as sitk
import numpy as np
import torch
import random
from skimage.metrics import hausdorff_distance
def get_samples_per_epoch(N_sample, sample_times,init_percent, max_percent):
        
        init_sample = int(init_percent * N_sample)
        sample_strategy = np.zeros((sample_times+1,))
        sample_strategy[0] = init_sample

        n_sample_num = int(max_percent * N_sample - init_sample)
        first_sample = min(int(n_sample_num/(sample_times//2)),init_sample)
        sample_strategy[1] = first_sample

        
        d = n_sample_num // sample_times  
        for i in range(1,sample_times + 1):
            sample_strategy[i] = d
        return sample_strategy

def random_sampling(sample_pool=[],init_percent=0.1):
        assert len(sample_pool) != 0

        # Group samples by patient ID
        patient_samples = {}
        for sample in sample_pool:
            patient_id = os.path.basename(sample).split('_')[0]  # Extract patient ID from file name  segthor->1other->0
            if patient_id not in patient_samples:
                patient_samples[patient_id] = []
            patient_samples[patient_id].append(sample)
        patient_ids = []
        for path in sample_pool:
            basename = os.path.basename(path)  # 获取文件名
            import re
            match = re.search(r'patient\d+', basename)  # 匹配patientxxx
            if match:
                patient_ids.append(match.group())  # 如果匹配成功，添加到列表中
        unique_patient_ids = list(set(patient_ids))
        unique_patient_ids.sort()
        random.seed(100)
        patient_ids = random.sample(unique_patient_ids, k=int(len(unique_patient_ids) * init_percent))  #没有主动学习，改成max_percent
        samples = []
        for patient_id in patient_ids:
            if patient_id in patient_samples:
                samples += patient_samples[patient_id]
        return samples

def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image