# -*- coding: utf-8 -*-

"""
@author: JiangSu
Email: jiangsukust@163.com

"""

import scipy.io as sio
from pypls import *
import os

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ================== Load Spectra DataSet ==================
# https://eigenvector.com/resources/data-sets/
# This data set consists of 80 samples of corn measured on 3 different NIR spectrometers.
# The wavelength range is 1100-2498nm at 2 nm intervals (700 channels).
# The moisture, oil, protein and starch values for each of the samples is also included.
# A number of NBS glass standards were also measured on each instrument.
# The data was originally taken at Cargill.

mat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'Benchmark_Corn.mat')
data = sio.loadmat(mat_path)

# the only difference between 'spectra' and 'absorbance' is that
# 'spectra' array includes wavelength in the first row.

# ------ take M5 for example ------
spectra = data['m5']  # (81, 700)
absorbance = spectra[1:, :]  # (80, 700)   80 samples's nir spectra, 700 data points
# ------ take 'oil' for example ------
target = data['oil']  # (80, 1) concentration of Oil
wavelength = spectra[0, :]  # 1100 - 2498 nm



# ================== samples split(Kennard-Stone) ==================
# divide the total samples into 3 parts: calset, valset, testset
calset_indices, valset_indices, testset_indices = samples_ks_split(absorbance, val_size=0.2, test_size=0.2)
xcal = np.vstack((wavelength, absorbance[calset_indices, :]))
ycal = target[calset_indices]
xval = np.vstack((wavelength, absorbance[valset_indices, :]))
yval = target[valset_indices]
xtest = np.vstack((wavelength, absorbance[testset_indices, :]))
ytest = target[testset_indices]


# ================== start computing ==================
# Note: In PartialLeastSquares Class, all X inputs should include 'wavelength'
# 'pretreat_method2' include 'MC' and 'ZS'
pls_instance = PartialLeastSquares(algorithm='ikpls_algorithm',
                                   max_nlv=20,
                                   pretreat_method1='SGMSC',
                                   pretreat_params1={'window_size':5, 'deriv':1},
                                   pretreat_method2='MC',
                                   customized_regions=[[1100, 1500], [1600, 2400]])

# -------------- cross validation --------------
# includes: 'cv_result' and 'cal_result'
pls_cv_result = pls_instance.cv(xcal, ycal, cv_sampling_method='cv_kfold_systematic_sampling',
                                sampling_param={'kfold':10}, calset_indices=None)

# -------------- valset validation --------------
# includes: 'vv_result' and 'cal_result'
pls_vv_result = pls_instance.vv(xcal, ycal, xval, yval)

# -------------- prediction  --------------
# You can specify the 'nlv' yourself, or using the 'optimal_nlv' generated from 'cv' or 'vv'
pls_predict_result = pls_instance.predict(xtest, nlv=None, testset_indices=testset_indices, testset_target=ytest)



print('Thank you!')
