# -*- coding: utf-8 -*-

"""
@author: JiangSu
Email: jiangsukust@163.com

"""

import scipy.io as sio
from pypls import *

# ++++++++++++++++++++++++++++ 导入数据 & 建立模型 ++++++++++++++++++++++++++++++++++++++++++
# ================== step1: Load DataSet ==================
data = sio.loadmat(r'Diesel_without_outlier.mat')
# -------------- 以总芳烃 total_aromatics 为例 --------------
xcal = data['total_aromatics_xcal']  # 校正集波长与吸光度，第一行为波长 (119, 401)
wavelength = xcal[1, :]
ab_cal = xcal[1:, :]  # 校正集吸光度
ycal = data['total_aromatics_ycal'].ravel()  # 校正集参考值 (118,)
xtest = data['total_aromatics_xtest']   # 测试集波长与吸光度，第一行为波长 (119, 401)
ab_test = xtest[1:, :]  # 测试集吸光度
ytest = data['total_aromatics_ytest'].ravel()  # 测试集参考值 (118,)

# -------------- 总芳烃含量统计结果 --------------
cal_min = ycal.min()
cal_max = ycal.max()
test_min = ytest.min()
test_max = ytest.max()
cal_range = cal_max - cal_min
test_range = test_max - test_min
cal_mean = ycal.mean()
test_mean = ytest.mean()
cal_std = np.std(ycal, ddof=1)
test_std = np.std(ytest, ddof=1)


# ================== step2: Build Model through Cross Validation ==================
# Note: In PartialLeastSquares Class, all X inputs should include 'wavelength'
# 'pretreat_method2' include 'MC' and 'ZS'
# -------------- PartialLeastSquares 类初始化 --------------
pls_instance = PartialLeastSquares(algorithm='simpls_algorithm',
                                   max_nlv=20,
                                   pretreat_method1=None,
                                   pretreat_params1=None,
                                   pretreat_method2='MC',
                                   customized_regions=None)
# -------------- cross validation --------------
# returned results include: 'cv_result' and 'cal_result'
pls_cv_result = pls_instance.cv(xcal,
                                ycal,
                                cv_sampling_method='cv_kfold_systematic_sampling',
                                sampling_param={'kfold':10})

# ---- 获取最佳因子数 ----
optimal_nlv = pls_cv_result['cv_result']['optimal_nlv']
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# ++++++++++++++++++++++++++++ 预测全部测试集样本 ++++++++++++++++++++++++++++++++++++++++++
# ================== Predict All Test Samples ==================
# You can specify the 'nlv' yourself, or using the 'optimal_nlv' generated from 'cv' or 'vv'
pls_predict_result = pls_instance.predict(testset_spec_intersect=xtest,
                                          nlv=None,
                                          testset_indices=None,
                                          testset_target=ytest)
# pls_predict_result即全范围的结果
range_full_predict_result = pls_predict_result
range_full_ytest = ytest
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# ++++++++++++++++++++++++++++ 演示测试集样本参考值方差对决定系数的影响 ++++++++++++++++++++++++++
# -------------- step1: 首先编写一个从一维数组正中间取k个元素的函数 --------------
def find_middle_k_elements(dim1_array, k):
    try:
        k = np.abs(int(k))
    except:
        raise TypeError("k have to be of type int")
    if len(dim1_array) == 0:
        raise ValueError("empty array")
    if len(dim1_array) < k:
        raise ValueError("n must less than the array's length")
    if k == 1:
        raise ValueError("are u kidding me?")

    # Get the length of the array
    length = len(dim1_array)

    # If the dim1_array's length is odd
    if length % 2 != 0:
        middle_index = length // 2
        start_ind = middle_index - k // 2
        return dim1_array[start_ind: start_ind + k]
    # If the dim1_array's length is even
    else:
        first_middle_index = length // 2 - 1
        second_middle_index = length // 2
        start_ind = first_middle_index - k // 2 + 1
        if k == 2:
            return dim1_array[first_middle_index: second_middle_index + 1]
        else:
            return dim1_array[start_ind: start_ind + k]

# -------------- step2: 获取ytest按参考值从小到大排序的索引号 --------------
sort_indices = np.argsort(ytest)


# -------------- Range_1 取中间30个样本 --------------
range_1_indices = find_middle_k_elements(sort_indices, 30)
range_1_ab_test = ab_test[range_1_indices, :]
range_1_spec_test = np.vstack((wavelength, range_1_ab_test))  # 加上波长，组装成光谱数组
range_1_ytest = ytest[range_1_indices]

# -------------- Range_2 取中间55个样本 --------------
range_2_indices = find_middle_k_elements(sort_indices, 55)
range_2_ab_test = ab_test[range_2_indices, :]
range_2_spec_test = np.vstack((wavelength, range_2_ab_test))  # 加上波长，组装成光谱数组
range_2_ytest = ytest[range_2_indices]

# -------------- Range_3 取中间75个样本 --------------
range_3_indices = find_middle_k_elements(sort_indices, 75)
range_3_ab_test = ab_test[range_3_indices, :]
range_3_spec_test = np.vstack((wavelength, range_3_ab_test))  # 加上波长，组装成光谱数组
range_3_ytest = ytest[range_3_indices]

# -------------- 预测 Range_1样本 --------------
range_1_predict_result = pls_instance.predict(testset_spec_intersect = range_1_spec_test,
                                              nlv=None,
                                              testset_indices=None,
                                              testset_target=range_1_ytest)

# -------------- 预测 Range_2样本 --------------
range_2_predict_result = pls_instance.predict(testset_spec_intersect = range_2_spec_test,
                                              nlv=None,
                                              testset_indices=None,
                                              testset_target=range_2_ytest)

# -------------- 预测 Range_3样本 --------------
range_3_predict_result = pls_instance.predict(testset_spec_intersect = range_3_spec_test,
                                              nlv=None,
                                              testset_indices=None,
                                              testset_target=range_3_ytest)

# -------------- 统计结果 --------------
r2_range_1 = range_1_predict_result['r2']
r2_range_2 = range_2_predict_result['r2']
r2_range_3 = range_3_predict_result['r2']
r2_range_full = range_full_predict_result['r2']

sse_range_1 = np.sum((range_1_predict_result['predict_value'].ravel() - range_1_ytest) ** 2)
sse_range_2 = np.sum((range_2_predict_result['predict_value'].ravel() - range_2_ytest) ** 2)
sse_range_3 = np.sum((range_3_predict_result['predict_value'].ravel() - range_3_ytest) ** 2)
sse_range_full = np.sum((range_full_predict_result['predict_value'].ravel() - range_full_ytest) ** 2)

sst_range_1 = np.sum((range_1_ytest - range_1_ytest.mean()) ** 2)
sst_range_2 = np.sum((range_2_ytest - range_2_ytest.mean()) ** 2)
sst_range_3 = np.sum((range_3_ytest - range_3_ytest.mean()) ** 2)
sst_range_full = np.sum((range_full_ytest - range_full_ytest.mean()) ** 2)

rmsep_range_1 = range_1_predict_result['rmsep']
rmsep_range_2 = range_2_predict_result['rmsep']
rmsep_range_3 = range_3_predict_result['rmsep']
rmsep_range_full = range_full_predict_result['rmsep']
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# ++++++++++++++++++++++++++++ 在Range_1中加入两个样本(最小值和最大值) ++++++++++++++++++++++++++
min_index = sort_indices[0]
max_index = sort_indices[-1]

range_1_modified_indices = np.concatenate((range_1_indices, np.array([min_index, max_index])))
range_1_modified_ab_test = ab_test[range_1_modified_indices, :]
range_1_modified_spec_test = np.vstack((wavelength, range_1_modified_ab_test))  # 加上波长，组装成光谱数组
range_1_modified_ytest = ytest[range_1_modified_indices]

# -------------- 预测 Range_1_modified样本 --------------
range_1_modified_predict_result = pls_instance.predict(testset_spec_intersect = range_1_modified_spec_test,
                                                       nlv=None,
                                                       testset_indices=None,
                                                       testset_target=range_1_modified_ytest)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# ++++++++++++++++++++++++++++ 其他统计量 ++++++++++++++++++++++++++
mae_range_1 = np.sum(np.abs(range_1_predict_result['predict_value'].ravel() - range_1_ytest)) / len(range_1_ytest)
mae_range_2 = np.sum(np.abs(range_2_predict_result['predict_value'].ravel() - range_2_ytest)) / len(range_2_ytest)
mae_range_3 = np.sum(np.abs(range_3_predict_result['predict_value'].ravel() - range_3_ytest)) / len(range_3_ytest)
mae_range_full = np.sum(np.abs(range_full_predict_result['predict_value'].ravel() - range_full_ytest)) / len(range_full_ytest)
mae_range_1_modified = np.sum(np.abs(range_1_modified_predict_result['predict_value'].ravel() - range_1_modified_ytest)) / len(range_1_modified_ytest)

mre_range_1 = np.mean(np.abs(range_1_predict_result['predict_value'].ravel() - range_1_ytest) / range_1_ytest)
mre_range_2 = np.mean(np.abs(range_2_predict_result['predict_value'].ravel() - range_2_ytest) / range_2_ytest)
mre_range_3 = np.mean(np.abs(range_3_predict_result['predict_value'].ravel() - range_3_ytest) / range_3_ytest)
mre_range_full = np.mean(np.abs(range_full_predict_result['predict_value'].ravel() - range_full_ytest) / range_full_ytest)
mre_range_1_modified = np.mean(np.abs(range_1_modified_predict_result['predict_value'].ravel() - range_1_modified_ytest) / range_1_modified_ytest)




print('Thank you!')

