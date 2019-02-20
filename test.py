# -*- coding: utf-8 -*-

"""
@author: JiangSu
Email: jiangsukust@163.com

"""
from SpectraAnalysis.pypls import *
import scipy.io as sio


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ================== Load Spectra DataSet ==================
data = sio.loadmat(r'Thermo_NIR_DataSet.mat')
# the only difference between 'spectra' and 'absorbance' is that 'spectra' array includes wavelength in the first row
spectra = data['spec']  # (622, 1557)
absorbance = data['ab']  # (621, 1557) 621 samples's nir spectra, 1557 data points
target = data['con']  # (621, 1) concentration of Nicotine in tobacco leaf
wavelength = spectra[0, :]  # 10001 - 3999 cm-1

# ================== samples split(Kennard-Stone) ==================
# divide the total samples to 3 parts: calset, valset, testset
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
                                   pretreat_method1='MSCSG',
                                   pretreat_params1={'window_size':5, 'deriv':1},
                                   pretreat_method2='ZS',
                                   customized_regions=[[4000, 6000], [8000, 10000]])

# -------------- cross validation --------------
# includes: 'cv_result' and 'cal_result'
pls_cv_result = pls_instance.cv(xcal, ycal, cv_sampling_method='cv_lpo_systematic_sampling',
                                sampling_param={'p':30}, calset_indices=None)

# -------------- valset validation --------------
# includes: 'vv_result' and 'cal_result'
pls_vv_result = pls_instance.vv(xcal, ycal, xval, yval)

# -------------- prediction  --------------
# You can specify the 'nlv' yourself, or using the 'optimal_nlv' generated from 'cv' or 'vv'
pls_predict_result = pls_instance.predict(xtest, nlv=None, testset_indices=testset_indices, testset_target=ytest)


print('Thank you!')