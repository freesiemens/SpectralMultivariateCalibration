# -*- coding: utf-8 -*-

"""
@author: JiangSu
Email: jiangsukust@163.com

============================= pypls =============================

Provides
  1. PartialLeastSquares(CrossValidation, ValsetValidation, Prediction)
     -- CrossValidation, cv
     -- ValsetValidation, vv
     -- Prediction, predict
     ++ It should be pointed out that before using 'predict', 'cv' or 'vv' must be run first.

     Take 'cv' for example, its outputs include 'cv_result' and 'cal_result'.

     Assume the DataSet consists of 80 spectra, 700 wavelength points, the max num of latent variable is 5.

     The cal_result's outputs are as following:

     'cal_result' including:
        'b': (回归系数，(700,5))
        't2_limit': (t2阈值，(6,5))
        'leverage_limit': (杠杆值阈值，(5,))
        'q_limit': (Q残差阈值，(6,5)，最后一列nan)
        't_critical_value': (y学生化残差阈值，(6,5))
        'r2': (决定系数R2，(5,))
        'press': (预测残差平方和，(5,))
        'rmsec': (RMSEC校正均方根误差，(5,))
        'sec': (SEC校正标准偏差，(5,))
        'rpd': (RPD，(5,))
        'bias': (Bias，(5,))
        'x_loadings': (X载荷，(700,5))
        'x_scores_weights': (X权重，(700,5))
        'linear_regression_coefficient': (包含斜率Slope和截距Offset，(2,5))
        'fitting_x_list': (list, 每个元素代表1个隐变量下的拟合光谱矩阵)
        'residual_matrix_list': (list, 每个元素代表1个隐变量下的残差光谱矩阵)
        'fit_value': (拟合值，(80,5))
        'y_residual': (拟合残差，(80,5))
        'x_residual': (X残差，(80,5))
        't2': (T2，(80,5))
        'leverage': (Leverage，(80,5))
        'x_scores': (X得分，(80,5))
        'x_fvalue': (X残差F分布统计量，(80,5))
        'x_fprob': (X残差F分布累积概率值，(80,5))
        'y_fvalue': (y残差F分布统计量，(80,5))
        'y_fprob': (y残差F分布累积概率值，(80,5))
        'y_tvalue': (y学生化残差，(80,5))  # 学生化残差
        'x_sample_residuals': (80,5)
        'x_variable_residuals': (700,5)
        'x_total_residuals': (1,5)
        'explained_x_sample_variance': (80,5)
        'explained_x_variable_variance': (700,5)
        'explained_x_total_variance': (1,5)
        'explained_x_variance_ratio': (1,5)
        'x_outlier_indices_list':
        'y_outlier_indices_list':
        'just_x_outlier_list':
        'just_y_outlier_list':
        'both_xy_outlier_list':

  2. Three PLS Algorithm:
     -- Improved Kernel Partial Least Squares, IKPLS
     -- Nonlinear Iterative Partial Least Squares，NIPALS
     -- Straightforward Implementation of a statistically inspired Modification of the Partial Least Squares, SIMPLS

  3. Several Sampling Algorithm:
     -- montecarlo_sampling
     -- ks_sampling(Kennard-Stone)
     -- spxy_sampling

  4. Several Samples split Algorithm:
     -- samples_systematic_split
     -- samples_ks_split
     -- samples_spxy_split
     -- samples_random_split

  5. Popular Pretreat methods for Spectroscopy
     -- Multiplicative Scatter Correction 多元散射校正, MSC
     -- Multiplicative Scatter Correction + Savitzky-Golay 多元散射校正+求导, MSCSG
     -- Vector Normalization 矢量归一化, VN
     -- Standard Normal Variate transformation 标准正态变换, SNV
     -- Eliminate Constant Offset 消除常数偏移量, ECO
     -- Subtract Straight Line 减去一条直线, SSL
     -- De-Trending 去趋势, DT
     -- Min-Max Normalization 最小最大归一化, MMN
     -- Savitzky-Golay 平滑与求导, SG
     -- SNV + Savitzky-Golay, SNVSG
     -- SNV + DT, SNVDT
     -- SSL + SG, SSLSG
     -- Mean Centering 均值中心化, MC
     -- Zscore Standardization 标准化, ZS

"""

import numpy as np
from numpy import diag, cumsum, where, dot, outer, zeros, sqrt, mean, sum, min, square, inner
from numpy.linalg import inv, norm
import scipy.stats as sps
from scipy.spatial.distance import pdist, squareform

# ================ PartialLeastSquares Class (Main)================
class PartialLeastSquares(object):
    '''
    Including 3 important functions, which are 'cv'(CrossValidation), 'vv'(ValsetValidation) and 'predict'(Prediction).
    It should be pointed out that before 'predict', 'cv' or 'vv' must be run first.
    '''
    def __init__(self,
                 algorithm='ikpls_algorithm',
                 max_nlv=10,
                 pretreat_method1='SG',
                 pretreat_params1=None,
                 pretreat_method2='MC',
                 customized_regions=[[4000,6000], [5000, 8000]]
                 ):
        self.algorithm = algorithm
        self.max_nlv = max_nlv
        self.pretreat_method1 = pretreat_method1
        if pretreat_params1 is None:
            self.pretreat_params1 = {}
        else:
            self.pretreat_params1 = pretreat_params1
        if pretreat_method1 is None:
            self.pretreat_params1 = {}
        self.pretreat_method2 = pretreat_method2
        self.customized_regions = customized_regions
        self.significance_level = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25]

        return

    def _sec_calc(self, fit_value, reference_value):
        '''
        只能用于Calibration时，拟合值与参考值。只能用于实例内部
        不能用于交叉验证
        :param fit_value:
        :param reference_value:
        :param nlv:
        :param ifmc:
        :return:
        '''
        if fit_value.ndim == 1:
            fit_value = fit_value[:, np.newaxis]  # 如果一维数组，增加至二维数组
        if reference_value.ndim == 1:
            reference_value = reference_value[:, np.newaxis]  # 如果一维数组，增加至二维数组
        max_nlv = fit_value.shape[1]
        n_samples = reference_value.shape[0]
        error = fit_value - reference_value
        # Error Sum of Squares(SSE)
        press = np.sum(error * error, axis=0)
        rmsec = sqrt(press / n_samples)
        # Total Sum Of Squares(SST) 总离差平方和
        sst = np.sum((reference_value - mean(reference_value)) ** 2)
        # Regression Sum of Squares(SSR)  (1, 10)
        ssr = np.sum((fit_value - mean(reference_value)) ** 2, axis=0)
        # SST = SSR + SSE
        # r2 = ssr / sst = 1 - sse / sst
        # r2 = ssr / sst
        r2 = 1 - press / sst

        sd = sqrt(sst / (n_samples - 1))  # 参考值的标准偏差
        # sd = np.std(reference_value, axis=0, ddof=1)
        bias = np.mean(error, axis=0)

        # ------------- 数据线性回归(横坐标reference_value, 纵坐标fit_value)
        # linear_regression_coefficient (2, max_nlv)  slope, intercept
        linear_regression_coefficient = zeros((2, max_nlv))

        # -------------- 校正标准误差 SEC (Standard Error of Calibration, 与自由度有关)
        sec = zeros(self.max_nlv)
        for i in range(self.max_nlv):
            nlv = i + 1
            if self.pretreat_method2 is not None:
                df = n_samples - nlv - 1
            else:
                df = n_samples - nlv
            e = error[:, i]
            sec_lv = sqrt(np.sum(e * e, axis=0) / df)
            sec[i] = sec_lv

            reg_coeff = lsr(reference_value, fit_value[:, i], order=1)['regression_coefficient']
            linear_regression_coefficient[:, i] = reg_coeff.ravel()

        # ------------ 预测标准误差 SEP (Standard Error of Prediction)  refer to OPUS, User friendly
        SEP = sqrt((np.sum((error - bias) * (error - bias), axis=0)) / (n_samples - 1))
        rpd = sd / SEP

        relative_error = np.abs(error) / reference_value

        return {'r2': r2, 'rmsec': rmsec, 'sep': SEP, 'sec': sec, 'press': press, 'rpd': rpd, 'bias': bias,
                'linear_regression_coefficient': linear_regression_coefficient,
                'relative_error': relative_error}

    def _spec_target_pretreat(self, spec, target):
        # ----------------- pretreat1 -----------------
        if self.pretreat_method1 is not None:
            self.pretreat4spec1 = eval(self.pretreat_method1.upper())(**self.pretreat_params1)  # 类的实例
            spec_pretreated1 = self.pretreat4spec1.fit_transform(spec)
        else:
            self.pretreat4spec1 = 'None'
            spec_pretreated1 = spec
        # 保存pretreat1预处理完的光谱的mean和stdev 为未知样本的pretreat2预处理做准备
        self.calx_pretreated1_mean = np.mean(spec_pretreated1[1:, :], axis=0)
        self.calx_pretreated1_stdev = np.std(spec_pretreated1[1:, :] - self.calx_pretreated1_mean, axis=0, ddof=1)
        # 保存y的mean和stdev 
        self.caly_mean = np.mean(target, axis=0)
        caly_mc = target - self.caly_mean
        self.caly_stdev = np.std(caly_mc, axis=0, ddof=1)

        # ----------------- pretreat2 -----------------
        if self.pretreat_method2 is not None:
            self.pretreat4spec2 = eval(self.pretreat_method2.upper())()
            self.pretreat4target = eval(self.pretreat_method2.upper() + '4Data')()
            spec_pretreated2 = self.pretreat4spec2.fit_transform(spec_pretreated1)
            target_pretreated = self.pretreat4target.fit_transform(target)
        else:
            self.pretreat4spec2 = 'None'
            self.pretreat4target = 'None'
            spec_pretreated2 = spec_pretreated1
            target_pretreated = target

        return spec_pretreated2, target_pretreated

    def _spec_pretreat4transform(self, spec_matrix):
        if self.pretreat_method1 is not None:
            # pretreat4spec1 ---- eval(self.pretreat_method1)(**self.pretreat_params1)
            # 实例对象
            spec_pretreated1 = self.pretreat4spec1.transform(spec_matrix)
        else:
            spec_pretreated1 = spec_matrix
        if self.pretreat_method2 is not None:
            # pretreat4spec2 ---- eval(self.pretreat_method2)(**self.pretreat_params2)
            # 实例对象
            spec_pretreated2 = self.pretreat4spec2.transform(spec_pretreated1)
        else:
            spec_pretreated2 = spec_pretreated1

        return spec_pretreated2

    def _target_inverse_pretreat(self, target):
        if self.pretreat_method2 is not None:
            target_inverse_pretreated = self.pretreat4target.inverse_transform(target)
        else:
            target_inverse_pretreated = target

        return target_inverse_pretreated

    def _spec_target_pretreat_cv(self, spec_cv, target_cv):
        # ----------------- pretreat1 -----------------
        if self.pretreat_method1 is not None:
            self.pretreat4spec1_cv = eval(self.pretreat_method1.upper())(**self.pretreat_params1)  # 类的实例
            spec_pretreated1_cv = self.pretreat4spec1_cv.fit_transform(spec_cv)
        else:
            self.pretreat4spec1_cv = 'None'
            spec_pretreated1_cv = spec_cv
        # 保存pretreat1预处理完的光谱的mean和stdev 
        self.calx_pretreated1_mean_cv = np.mean(spec_pretreated1_cv[1:, :], axis=0)
        self.calx_pretreated1_stdev_cv = np.std(spec_pretreated1_cv[1:, :] - self.calx_pretreated1_mean_cv, axis=0, ddof=1)
        # 保存y的mean和stdev 
        self.caly_mean_cv = np.mean(target_cv, axis=0)
        caly_mc_cv = target_cv - self.caly_mean_cv
        self.caly_stdev_cv = np.std(caly_mc_cv, axis=0, ddof=1)

        # ----------------- pretreat2 -----------------
        if self.pretreat_method2 is not None:
            self.pretreat4spec2_cv = eval(self.pretreat_method2.upper())()
            self.pretreat4target_cv = eval(self.pretreat_method2.upper() + '4Data')()
            spec_pretreated2_cv = self.pretreat4spec2_cv.fit_transform(spec_pretreated1_cv)
            target_pretreated_cv = self.pretreat4target_cv.fit_transform(target_cv)
        else:
            self.pretreat4spec2_cv = 'None'
            self.pretreat4target_cv = 'None'
            spec_pretreated2_cv = spec_pretreated1_cv
            target_pretreated_cv = target_cv

        return spec_pretreated2_cv, target_pretreated_cv

    def _spec_pretreat4transform_cv(self, spec_matrix_cv):
        if self.pretreat_method1 is not None:
            # pretreat4spec1 ---- eval(self.pretreat_method1)(**self.pretreat_params1)
            # 实例对象
            spec_pretreated1_cv = self.pretreat4spec1_cv.transform(spec_matrix_cv)
        else:
            spec_pretreated1_cv = spec_matrix_cv
        if self.pretreat_method2 is not None:
            # pretreat4spec2 ---- eval(self.pretreat_method2)(**self.pretreat_params2)
            # 实例对象
            spec_pretreated2_cv = self.pretreat4spec2_cv.transform(spec_pretreated1_cv)
        else:
            spec_pretreated2_cv = spec_pretreated1_cv

        return spec_pretreated2_cv

    def _target_inverse_pretreat_cv(self, target_cv):
        if self.pretreat_method2 is not None:
            target_inverse_pretreated_cv = self.pretreat4target_cv.inverse_transform(target_cv)
        else:
            target_inverse_pretreated_cv = target_cv

        return target_inverse_pretreated_cv

    def calibration(self, calset_spec_intersect, calset_target, calset_indices=None):
        if calset_target.ndim == 1:
            calset_target = calset_target[:, np.newaxis]  # 用于多维结果的broadcast计算
        calset_ab_intersect = calset_spec_intersect[1:, :]
        self.calx_mean = np.mean(calset_ab_intersect, axis=0)
        n_samples = calset_ab_intersect.shape[0]
        if calset_indices is None:
            calset_indices = np.arange(n_samples)

        # ------------- 光谱预处理，浓度预处理 -------------
        spec, target = self._spec_target_pretreat(calset_spec_intersect, calset_target)
        # ------------- 截取波长点 -------------
        spec_subset = spec[:, self.variable_indices]
        ab_subset = spec_subset[1:, :]
        # ------------- 开始PLSR -------------
        calplsr = PLSR(self.algorithm, self.max_nlv)
        cal_result = calplsr.fit_predict(spec_subset, target)
        b = cal_result['b']
        fit_value_temp = cal_result['fit_value']
        fit_value = self._target_inverse_pretreat(fit_value_temp)
        pls_result = cal_result['pls_result']
        x_loadings = cal_result['x_loadings']
        x_scores = cal_result['x_scores']
        x_scores_weights = cal_result['x_scores_weights']

        # ------------- 开始统计指标计算 -------------
        # -------- 光谱残差
        q_result = q_calc(x_loadings, x_scores, ab_subset)
        q = q_result['q']
        x_residual = sqrt(q_result['q'])
        residual_matrix_list = q_result['residual_matrix_list']
        fitting_x_list = q_result['fitting_x_list']
        x_sample_residuals = q_result['x_sample_residuals']
        x_variable_residuals = q_result['x_variable_residuals']
        x_total_residuals = q_result['x_total_residuals']
        explained_x_sample_variance = q_result['explained_x_sample_variance']
        explained_x_variable_variance = q_result['explained_x_variable_variance']
        explained_x_total_variance = q_result['explained_x_total_variance']
        explained_x_variance_ratio = q_result['explained_x_variance_ratio']

        # ---- 计算t2_limit, t_critical_value, q_limit
        sl = self.significance_level  # 显著性水平
        t2_limit = zeros((len(sl), self.max_nlv))
        t_critical_value = zeros((len(sl), self.max_nlv))  # y_tvalue 学生化残差的临界值

        q_limit = zeros((len(sl), self.max_nlv))
        # refer to: Interpreting PLS plots
        # The critical value of the Q-residuals are estimated from the eigenvalues of E, as described in Jackson and Mudholkar, 1979.
        prevent_invalid_for_nan_warn = np.seterr(invalid='ignore')  # 最后一个隐变量的q_limit无法计算
        eigenvalues_list = []
        for lv in range(self.max_nlv):
            e = q_result['residual_matrix_list'][lv]
            U, S, V = np.linalg.svd(e, full_matrices=False)
            eigenvalues_list.append(S ** 2 / (n_samples - 1))  # note the (n_samples - 1) part for unbiased estimate of var

        for i in range(self.max_nlv):  # 0:5 nlv
            for j in range(len(sl)):  # 不同显著性水平
                # ---- t2_limit ----
                nlv = i + 1
                # .ppf的参数 q ---- lower tail probability
                t2_limit_sl = nlv * (n_samples - 1) / (n_samples - nlv) * sps.f.ppf(1 - sl[j], nlv, n_samples - nlv)
                t2_limit[j, i] = t2_limit_sl

                # ---- 学生化残差临界值, t_critical_value双尾t检验, t.ppf((1 - sl[j]) / 2, df)
                if self.pretreat_method2 is not None:
                    df = n_samples - nlv - 1
                else:
                    df = n_samples - nlv
                t_critical_value_sl = sps.t.ppf(1 - sl[j] / 2, df)
                t_critical_value[j, i] = t_critical_value_sl

                # ---- q_limit ----
                evalues_unused = eigenvalues_list[i]
                theta1 = np.sum(evalues_unused)
                theta2 = np.sum(evalues_unused ** 2)
                theta3 = np.sum(evalues_unused ** 3)
                h0 = 1 - (2 * theta1 * theta3) / (3 * theta2 ** 2)
                if h0 < 0.001:
                    h0 = 0.001
                # .ppf的参数 q ---- lower tail probability
                ca = sps.norm.ppf(1 - sl[j])
                h1 = ca * sqrt(2 * theta2 * h0 ** 2) / theta1
                h2 = theta2 * h0 * (h0 - 1) / (theta1 ** 2)
                # 不同显著性水平
                q_limit_sl = theta1 * (1 + h1 + h2) ** (1 / h0)
                q_limit[j, i] = q_limit_sl

        # -------- Leverage & Hotelling TSquared
        leverage_t2_result = leverage_t2_calc(x_scores, x_scores)
        leverage = leverage_t2_result['leverage']
        leverage_limit = 3 * mean(leverage, axis=0)
        t2 = leverage_t2_result['t2']

        # x_Fvalue, x_Fprob ---- refer to OPUS
        x_fvalue = (n_samples - 1) * x_residual ** 2 / (sum(square(x_residual), axis=0) - x_residual ** 2)
        x_fprob = sps.distributions.f.cdf(x_fvalue, 1, n_samples - 1)
        # y_Fvalue, y_Fprob ---- refer to OPUS
        y_residual = fit_value - calset_target
        y_fvalue = (n_samples - 1) * y_residual ** 2 / (sum(square(y_residual), axis=0) - y_residual ** 2)
        y_fprob = sps.distributions.f.cdf(y_fvalue, 1, n_samples - 1)

        # 计算r2, SEC, press, rpd, bias(全部隐变量)
        sec_statistics_result = self._sec_calc(fit_value, calset_target)
        r2 = sec_statistics_result['r2']
        press = sec_statistics_result['press']
        rmsec = sec_statistics_result['rmsec']
        sec = sec_statistics_result['sep']  
        rpd = sec_statistics_result['rpd']
        bias = sec_statistics_result['bias']
        linear_regression_coefficient = sec_statistics_result['linear_regression_coefficient']
        relative_error = sec_statistics_result['relative_error']

        # ---- 20190115增加y_tvalue(学生化残差)
        prevent_invalid_for_negetive_sqrt = np.seterr(invalid='ignore')
        y_tvalue = y_residual / (rmsec * sqrt(1 - leverage))

        # ---- outlier detect
        outlier_dectect_result = outlier_detect(leverage, leverage_limit, y_fprob, calset_indices)
        x_outlier_indices_list = outlier_dectect_result['x_outlier_indices_list']
        y_outlier_indices_list = outlier_dectect_result['y_outlier_indices_list']
        just_x_outlier_list = outlier_dectect_result['just_x_outlier_list']
        just_y_outlier_list = outlier_dectect_result['just_y_outlier_list']
        both_xy_outlier_list = outlier_dectect_result['both_xy_outlier_list']

        ########################## 预测需要 ##########################
        # calx_mean, calx_pretreated1_mean, calx_pretreated1_stdev,
        # caly_mean, caly_stdev, b, calx_loadings, calx_scores, calx_scores_weights,
        # leverage_limit, testset_indices = None
        model_parameters = {'pretreat_method1':self.pretreat_method1,
                            'pretreat_params1':self.pretreat_params1,
                            'pretreat_method2':self.pretreat_method2,
                            'calx_mean':self.calx_mean,
                            'calx_pretreated1_mean':self.calx_pretreated1_mean,
                            'calx_pretreated1_stdev':self.calx_pretreated1_stdev,
                            'caly_mean':self.caly_mean,
                            'caly_stdev':self.caly_stdev,
                            'b':b,
                            'calx_loadings':x_loadings,
                            'calx_scores':x_scores,
                            'calx_scores_weights':x_scores_weights,
                            'leverage_limit':leverage_limit,
                            't2_limit':t2_limit,
                            'q_limit':q_limit,
                            'variable_indices':self.variable_indices}

        return {'b':b,
                'fit_value': fit_value,
                'y_residual': y_residual,
                'x_residual': x_residual,
                'fitting_x_list': fitting_x_list,
                'residual_matrix_list': residual_matrix_list,
                'x_sample_residuals': x_sample_residuals,
                'x_variable_residuals': x_variable_residuals,
                'x_total_residuals': x_total_residuals,
                'explained_x_sample_variance': explained_x_sample_variance,
                'explained_x_variable_variance': explained_x_variable_variance,
                'explained_x_total_variance': explained_x_total_variance,
                'explained_x_variance_ratio': explained_x_variance_ratio,
                'pls_result': pls_result,
                't2': t2,
                't2_limit':t2_limit,
                'leverage': leverage,
                'leverage_limit': leverage_limit,
                'q': q,
                'q_limit': q_limit,
                'x_loadings': x_loadings,
                'x_scores': x_scores,
                'x_scores_weights': x_scores_weights,
                'x_fvalue': x_fvalue,
                'x_fprob': x_fprob,
                'y_fvalue': y_fvalue,
                'y_fprob': y_fprob,
                'y_tvalue': y_tvalue,  # 学生化残差
                't_critical_value': t_critical_value,  # 学生化残差阈值
                'r2': r2,
                'press': press,
                'rmsec': rmsec,
                'sec': sec,
                'rpd': rpd,
                'bias':bias,
                'linear_regression_coefficient': linear_regression_coefficient,
                'relative_error': relative_error,
                'x_outlier_indices_list': x_outlier_indices_list,
                'y_outlier_indices_list': y_outlier_indices_list,
                'just_x_outlier_list': just_x_outlier_list,
                'just_y_outlier_list': just_y_outlier_list,
                'both_xy_outlier_list': both_xy_outlier_list,
                'model_parameters':model_parameters}

    def cv(self, calset_spec_intersect, calset_target, cv_sampling_method='cv_lpo_systematic_sampling',
           sampling_param={'p': 3}, calset_indices=None):
        '''
        Cross PLSValidation
        :return:
        '''
        if calset_target.ndim == 1:
            calset_target = calset_target[:, np.newaxis]  # 用于多维结果的broadcast计算
        self.cv_sampling_method = cv_sampling_method
        self.sampling_param=sampling_param
        self.calset_target = calset_target
        self.calset_spec_intersect = calset_spec_intersect
        self.calset_wavelength_intersect = self.calset_spec_intersect[0, :]
        self.calset_ab_intersect = self.calset_spec_intersect[1:, :]
        n_cal_samples = self.calset_spec_intersect.shape[0] - 1
        if calset_indices is None:
            calset_indices = np.arange(n_cal_samples)
        # -------- 处理variable_indices (indices针对intersect, 而非全谱) --------
        # 手工选择的谱区或BIPLS得到的谱区; 如果是离散的波长点，则事先已经得到
        if self.customized_regions is not None:
            self.verified_regions = verify_customized_regions(self.calset_wavelength_intersect, self.customized_regions)
            self.variable_indices = generate_variable_indices(self.calset_wavelength_intersect, self.customized_regions)
        else:
            self.customized_regions = [[self.calset_wavelength_intersect[0], self.calset_wavelength_intersect[-1]]]
            self.verified_regions = verify_customized_regions(self.calset_wavelength_intersect, self.customized_regions)
            self.variable_indices = generate_variable_indices(self.calset_wavelength_intersect, self.customized_regions)
        # 处理维数过大的问题
        if self.max_nlv > np.min((self.calset_spec_intersect.shape[0] - 1, self.variable_indices.size)):
            self.max_nlv = np.min((self.calset_spec_intersect.shape[0] - 1, self.variable_indices.size))

        n_variables = self.variable_indices.size
        # =========================== Calibration start ===========================
        self.cal_result = self.calibration(self.calset_spec_intersect, self.calset_target, calset_indices=calset_indices)
        calx_scores = self.cal_result['x_scores']
        leverage_limit = self.cal_result['leverage_limit']
        # =========================== Calibration end ===========================



        # =========================== Cross PLSValidation start ===========================
        x = self.calset_ab_intersect
        y = self.calset_target
        # -------- 交叉验证划分集合 --------
        train_indices_list, test_indices_list = eval(self.cv_sampling_method)(n_cal_samples, **self.sampling_param)
        n_fold = len(train_indices_list)
        cv_predict_value = zeros((n_cal_samples, self.max_nlv))  # 列出所有样本各个维数的预测结果
        cv_x_residual = zeros((n_cal_samples, self.max_nlv))  # 列出所有样本各个维数的光谱残差
        cv_y_residual = zeros((n_cal_samples, self.max_nlv))  # 列出所有样本各个维数的预测残差
        cv_x_scores = zeros((n_cal_samples, self.max_nlv))  # 列出所有样本各个维数的得分
        cv_residual_matrix = zeros((self.max_nlv, n_cal_samples, n_variables))  # 三维数组(nlv, m, n)
        cv_fitting_x = zeros((self.max_nlv, n_cal_samples, n_variables))  # 三维数组(nlv, m, n)
        cv_ab_pretreated = zeros((n_cal_samples, n_variables))  # 存储每个交叉验证中预处理后的样品
        cv_q = zeros((n_cal_samples, self.max_nlv))
        # ======================== 开始交叉验证 ========================
        for i in range(n_fold):
            calx_cv, caly_cv = x[train_indices_list[i], :], y[train_indices_list[i]]
            valx_cv, valy_cv = x[test_indices_list[i], :], y[test_indices_list[i]]
            # --------- 区分了校正子集和预测子集，开始光谱和组分预处理 ---------
            calspec_cv = np.vstack((self.calset_wavelength_intersect, calx_cv))
            valspec_cv = np.vstack((self.calset_wavelength_intersect, valx_cv))
            calspec_cv_pretreated, caly_cv_pretreated = self._spec_target_pretreat_cv(calspec_cv, caly_cv)
            valspec_cv_pretreated = self._spec_pretreat4transform_cv(valspec_cv)
            # --------- 根据variable indices, 截取波长点, 开始计算 ---------
            calspec_subset = calspec_cv_pretreated[:, self.variable_indices]
            valspec_subset = valspec_cv_pretreated[:, self.variable_indices]
            valx_subset = valspec_subset[1:, :]
            sub_plsr = PLSR(algorithm=self.algorithm, max_nlv=self.max_nlv)
            sub_plsr.fit(calspec_subset, caly_cv_pretreated)  # 只需要fit
            sub_pls_result = sub_plsr.pls_result
            calx_loadings_cv = sub_pls_result['x_loadings']
            # calx_scores_cv = sub_pls_result['x_scores']
            calx_scores_weights_cv = sub_pls_result['x_scores_weights']
            # --------- val_predict_value_temp 根据pretreat_method2做inverse_transform---------
            val_predicte_value_temp = sub_plsr.val_predict(valspec_subset)['predict_value']
            val_predict_value = self._target_inverse_pretreat_cv(val_predicte_value_temp)
            val_y_residual = val_predict_value - valy_cv
            cv_predict_value[test_indices_list[i], :] = val_predict_value
            # ---------- 部分统计指标 ----------
            val_x_scores = dot(valx_subset, calx_scores_weights_cv)
            cv_x_scores[test_indices_list[i], :] = val_x_scores
            # ---- 光谱残差
            cv_ab_pretreated[test_indices_list[i], :] = valx_subset  # 用于计算explained_x_total_variance
            val_q_result = q_calc_cv(calx_loadings_cv, val_x_scores, valx_subset)
            val_q = val_q_result['q']
            val_x_residual = sqrt(val_q_result['q'])
            val_residual_matrix_list = val_q_result['residual_matrix_list']  # 等待存储
            val_fitting_x_list = val_q_result['fitting_x_list']  # 等待存储
            for j in range((self.max_nlv)):  # 存入三维数组
                cv_residual_matrix[j, test_indices_list[i], :] = val_residual_matrix_list[j]
                cv_fitting_x[j, test_indices_list[i], :] = val_fitting_x_list[j]

            cv_q[test_indices_list[i], :] = val_q
            # ---- 处理 x_residual 与 y_residual
            cv_x_residual[test_indices_list[i], :] = val_x_residual
            cv_y_residual[test_indices_list[i], :] = val_y_residual

        # ---------------- 交叉验证完毕，统一计算 ----------------

        # ---- x_variable_residuals 和 x_total_residuals
        cv_x_sample_residuals = np.sum(cv_residual_matrix ** 2, axis=2).T / n_variables
        cv_x_variable_residuals = np.sum(cv_residual_matrix ** 2, axis=1).T / n_cal_samples  # (n_variables, n_lv)
        cv_x_total_residuals = np.mean(cv_x_variable_residuals, axis=0, keepdims=True)  # (1, n_lv)
        cv_explained_x_sample_variance = (1 - cv_x_sample_residuals / \
                                             (np.sum(cv_ab_pretreated ** 2, axis=1, keepdims=True) / n_variables)) * 100
        cv_explained_x_variable_variance = (1 - cv_x_variable_residuals.T /
                                            (np.sum(cv_ab_pretreated ** 2, axis=0) / n_cal_samples)) * 100
        cv_explained_x_total_variance = (1 - cv_x_total_residuals / np.mean(cv_ab_pretreated ** 2)) * 100
        cv_explained_x_variance_ratio = np.hstack((cv_explained_x_total_variance[:, 0:1],
                                                np.diff(cv_explained_x_total_variance)))

        # ---- 将 cv_residual_matrix 和 cv_fitting_x 三维数组重新处理成list
        cv_residual_matrix_list = [cv_residual_matrix[i, :, :] for i in range(self.max_nlv)]
        cv_fitting_x_list = [cv_fitting_x[i, :, :] for i in range(self.max_nlv)]


        # ---- Leverage & Hotelling TSquared  (20190120 OK)
        leverage_t2_result = leverage_t2_calc_cv(cv_x_scores, calx_scores)
        cv_leverage = leverage_t2_result['leverage']
        cv_t2 = leverage_t2_result['t2']

        # 计算x_residual的fvalue和fprob
        x_fvalue = (n_cal_samples - 1) * cv_x_residual ** 2 / (
            sum(square(cv_x_residual), axis=0) - cv_x_residual ** 2)
        x_fprob = sps.distributions.f.cdf(x_fvalue, 1, n_cal_samples - 1)
        # 计算y_residual的fvalue和fprob
        y_fvalue = (n_cal_samples - 1) * cv_y_residual ** 2 / (
            sum(square(cv_y_residual), axis=0) - cv_y_residual ** 2)
        y_fprob = sps.distributions.f.cdf(y_fvalue, 1, n_cal_samples - 1)
        # 计算r2, rmsecv, press, rpd, bias(全部维数)
        rmse_statistics = rmse_calc(cv_predict_value, self.calset_target)
        r2 = rmse_statistics['r2']
        rmsecv = rmse_statistics['rmse']
        secv = rmse_statistics['sep']
        press = rmse_statistics['press']
        rpd = rmse_statistics['rpd']
        bias = rmse_statistics['bias']
        linear_regression_coefficient = rmse_statistics['linear_regression_coefficient']
        relative_error = rmse_statistics['relative_error']

        # ---- 20190128增加y_tvalue(学生化残差)
        prevent_invalid_for_negetive_sqrt = np.seterr(invalid='ignore')
        y_tvalue = cv_y_residual / (rmsecv * sqrt(1 - cv_leverage))  # 20190128 与 Unscrambler 保持一致, 除以RMSECV

        # 推荐维数
        min_press = min(press)
        press_fvalue = press / min_press
        press_fprob = sps.distributions.f.cdf(press_fvalue, n_cal_samples, n_cal_samples)
        if np.all(press_fprob >= 0.75):
            self.optimal_nlv = self.max_nlv
        else:
            self.optimal_nlv = np.where(press_fprob < 0.75)[0][0] + 1
        optimal_rmsecv = rmsecv[self.optimal_nlv - 1]

        # ======== outlier 检测 ========
        outlier_dectect_result = outlier_detect(cv_leverage, leverage_limit, y_fprob, calset_indices)
        x_outlier_indices_list = outlier_dectect_result['x_outlier_indices_list']
        y_outlier_indices_list = outlier_dectect_result['y_outlier_indices_list']
        just_x_outlier_list = outlier_dectect_result['just_x_outlier_list']
        just_y_outlier_list = outlier_dectect_result['just_y_outlier_list']
        both_xy_outlier_list = outlier_dectect_result['both_xy_outlier_list']

        # ======== 保存cv结果 ========
        cv_result = {'predict_value': cv_predict_value,
                     'x_residual': cv_x_residual,
                     'y_residual': cv_y_residual,
                     'fitting_x_list': cv_fitting_x_list,
                     'residual_matrix_list': cv_residual_matrix_list,
                     'x_sample_residuals': cv_x_sample_residuals,
                     'x_variable_residuals': cv_x_variable_residuals,
                     'x_total_residuals': cv_x_total_residuals,
                     'explained_x_sample_variance': cv_explained_x_sample_variance,
                     'explained_x_variable_variance': cv_explained_x_variable_variance.T,
                     'explained_x_total_variance': cv_explained_x_total_variance,
                     'explained_x_variance_ratio': cv_explained_x_variance_ratio,
                     'leverage': cv_leverage,
                     't2': cv_t2,
                     'q': cv_q,
                     'x_scores': cv_x_scores,
                     'x_fvalue': x_fvalue,
                     'x_fprob': x_fprob,
                     'y_fvalue': y_fvalue,
                     'y_fprob': y_fprob,
                     'y_tvalue': y_tvalue,  # 学生化残差
                     'r2': r2,
                     'rmsecv': rmsecv,
                     'secv': secv,
                     'optimal_nlv': self.optimal_nlv,
                     'optimal_rmsecv': optimal_rmsecv,
                     'press': press,
                     'rpd': rpd,
                     'bias': bias,
                     'linear_regression_coefficient': linear_regression_coefficient,
                     'relative_error': relative_error,
                     'x_outlier_indices_list': x_outlier_indices_list,
                     'y_outlier_indices_list': y_outlier_indices_list,
                     'just_x_outlier_list': just_x_outlier_list,
                     'just_y_outlier_list': just_y_outlier_list,
                     'both_xy_outlier_list': both_xy_outlier_list}

        return {'cv_result': cv_result, 'cal_result': self.cal_result}

    def vv(self, calset_spec_intersect, calset_target, valset_spec_intersect, valset_target,
           calset_indices=None, valset_indices=None):
        '''
        Valset PLSValidation, 利用校正集校正，预测验证集，得出最佳nlv
        :return:
        '''
        if calset_target.ndim == 1:
            calset_target = calset_target[:, np.newaxis]  # 用于多维结果的broadcast计算
        if valset_target.ndim == 1:
            valset_target = valset_target[:, np.newaxis]  # 用于多维结果的broadcast计算
        self.calset_target = calset_target
        self.valset_target = valset_target
        self.calset_spec_intersect = calset_spec_intersect
        self.calset_wavelength_intersect = self.calset_spec_intersect[0, :]
        self.calset_ab_intersect = self.calset_spec_intersect[1:, :]
        self.valset_spec_intersect = valset_spec_intersect
        self.valset_wavelength_intersect = self.valset_spec_intersect[0, :]
        self.valset_ab_intersect = self.valset_spec_intersect[1:, :]
        # -------- 处理variable_indices (indices针对intersect, 而非全谱) --------
        if self.customized_regions is not None:
            self.verified_regions = verify_customized_regions(self.calset_wavelength_intersect, self.customized_regions)
            self.variable_indices = generate_variable_indices(self.calset_wavelength_intersect, self.customized_regions)
        else:
            self.customized_regions = [[self.calset_wavelength_intersect[0], self.calset_wavelength_intersect[-1]]]
            self.verified_regions = verify_customized_regions(self.calset_wavelength_intersect, self.customized_regions)
            self.variable_indices = generate_variable_indices(self.calset_wavelength_intersect, self.customized_regions)
        # 处理维数过大的问题
        if self.max_nlv > np.min((self.calset_spec_intersect.shape[0] - 1, self.variable_indices.size)):
            self.max_nlv = np.min((self.calset_spec_intersect.shape[0] - 1, self.variable_indices.size))

        # =========================== Calibration start ===========================
        self.cal_result = self.calibration(self.calset_spec_intersect, self.calset_target, calset_indices=calset_indices)
        leverage_limit = self.cal_result['leverage_limit']
        calx_loadings = self.cal_result['x_loadings']
        calx_scores = self.cal_result['x_scores']
        calx_scores_weights = self.cal_result['x_scores_weights']
        b = self.cal_result['b']
        # =========================== Calibration end ===========================


        # =========================== Valset PLSValidation start ===========================
        n_val_samples = self.valset_ab_intersect.shape[0]
        # ------------------  验证集光谱预处理 ------------------
        valspec_pretreated = self._spec_pretreat4transform(self.valset_spec_intersect)
        # --------- 根据variable indices, 截取波长点 ---------
        valspec_subset = valspec_pretreated[:, self.variable_indices]
        valx_subset = valspec_subset[1:, :]
        # --------- 开始预测 ---------
        val_predicte_value_temp = dot(valx_subset, b)
        val_predict_value = self._target_inverse_pretreat(val_predicte_value_temp)
        val_y_residual = val_predict_value - self.valset_target
        # ---------- 统计指标 ----------
        val_x_scores = dot(valx_subset, calx_scores_weights)

        # -------- Leverage & Hotelling TSquared
        leverage_t2_result = leverage_t2_calc(val_x_scores, calx_scores)
        val_leverage = leverage_t2_result['leverage']
        val_t2 = leverage_t2_result['t2']

        if n_val_samples < 2:
            # 保存vv结果
            vv_result = {'predict_value': val_predict_value,
                         'x_scores': val_x_scores,
                         'leverage': val_leverage,
                         't2': val_t2,
                         'y_residual': val_y_residual}
        else:
            # --------------- 验证完毕，统一计算 ---------------
            # ---- 光谱残差
            val_q_result = q_calc(calx_loadings, val_x_scores, valx_subset)
            val_q = val_q_result['q']
            val_f_residuals = val_q_result['f_residuals']
            val_x_residual = sqrt(val_q_result['q'])
            val_residual_matrix_list = val_q_result['residual_matrix_list']
            val_fitting_x_list = val_q_result['fitting_x_list']
            val_x_sample_residuals = val_q_result['x_sample_residuals']
            val_x_variable_residuals = val_q_result['x_variable_residuals']
            val_x_total_residuals = val_q_result['x_total_residuals']
            val_explained_x_sample_variance = val_q_result['explained_x_sample_variance']
            val_explained_x_variable_variance = val_q_result['explained_x_variable_variance']
            val_explained_x_total_variance = val_q_result['explained_x_total_variance']
            val_explained_x_variance_ratio = val_q_result['explained_x_variance_ratio']

            # ---- 计算x_residual的fvalue和fprob
            x_fvalue = (n_val_samples - 1) * val_x_residual ** 2 / (sum(square(val_x_residual), axis=0) - val_x_residual ** 2)
            x_fprob = sps.distributions.f.cdf(x_fvalue, 1, n_val_samples - 1)
            # 计算y_residual的fvalue和fprob
            y_fvalue = (n_val_samples - 1) * val_y_residual ** 2 / (sum(square(val_y_residual), axis=0) - val_y_residual ** 2)
            y_fprob = sps.distributions.f.cdf(y_fvalue, 1, n_val_samples - 1)
            # 计算r2, rmsecv, press, rpd, bias(全部维数)
            rmse_statistics = rmse_calc(val_predict_value, self.valset_target)
            r2 = rmse_statistics['r2']
            rmsep = rmse_statistics['rmse']
            sep = rmse_statistics['sep']
            press = rmse_statistics['press']
            rpd = rmse_statistics['rpd']
            bias = rmse_statistics['bias']
            linear_regression_coefficient = rmse_statistics['linear_regression_coefficient']
            relative_error = rmse_statistics['relative_error']

            # ---- 20190128增加y_tvalue(学生化残差)
            prevent_invalid_for_negetive_sqrt = np.seterr(invalid='ignore')
            y_tvalue = val_y_residual / (rmsep * sqrt(1 - val_leverage))  # 20190128 与 Unscrambler 保持一致, 除以RMSEP

            # 推荐维数
            min_press = min(press)
            press_fvalue = press / min_press
            press_fprob = sps.distributions.f.cdf(press_fvalue, n_val_samples, n_val_samples)
            if np.all(press_fprob >= 0.75) :
                self.optimal_nlv = self.max_nlv
            else:
                self.optimal_nlv = np.where(press_fprob < 0.75)[0][0] + 1
            optimal_rmsep = rmsep[self.optimal_nlv - 1]

            # ======== outlier 检测 ========
            outlier_dectect_result = outlier_detect(val_leverage, leverage_limit, y_fprob, valset_indices)
            x_outlier_indices_list = outlier_dectect_result['x_outlier_indices_list']
            y_outlier_indices_list = outlier_dectect_result['y_outlier_indices_list']
            just_x_outlier_list = outlier_dectect_result['just_x_outlier_list']
            just_y_outlier_list = outlier_dectect_result['just_y_outlier_list']
            both_xy_outlier_list = outlier_dectect_result['both_xy_outlier_list']

            # 保存vv结果
            vv_result = {'predict_value': val_predict_value,
                         'x_scores': val_x_scores,
                         'leverage': val_leverage,
                         't2': val_t2,
                         'q': val_q,
                         'val_f_residuals': val_f_residuals,
                         'y_residual': val_y_residual,
                         'x_residual': val_x_residual,
                         'fitting_x_list': val_fitting_x_list,
                         'residual_matrix_list': val_residual_matrix_list,
                         'x_sample_residuals': val_x_sample_residuals,
                         'x_variable_residuals': val_x_variable_residuals,
                         'x_total_residuals': val_x_total_residuals,
                         'explained_x_sample_variance': val_explained_x_sample_variance,
                         'explained_x_variable_variance': val_explained_x_variable_variance,
                         'explained_x_total_variance': val_explained_x_total_variance,
                         'explained_x_variance_ratio': val_explained_x_variance_ratio,
                         'x_fvalue': x_fvalue,
                         'x_fprob': x_fprob,
                         'y_fvalue': y_fvalue,
                         'y_fprob': y_fprob,
                         'y_tvalue': y_tvalue,  # 学生化残差
                         'r2': r2,
                         'rmsep': rmsep,
                         'sep': sep,
                         'optimal_nlv': self.optimal_nlv,
                         'optimal_rmsep': optimal_rmsep,
                         'press': press,
                         'rpd': rpd,
                         'bias': bias,
                         'linear_regression_coefficient': linear_regression_coefficient,
                         'relative_error': relative_error,
                         'x_outlier_indices_list': x_outlier_indices_list,
                         'y_outlier_indices_list': y_outlier_indices_list,
                         'just_x_outlier_list': just_x_outlier_list,
                         'just_y_outlier_list': just_y_outlier_list,
                         'both_xy_outlier_list': both_xy_outlier_list,
                         'residual_matrix_list': val_residual_matrix_list,
                         'fitting_x_list': val_fitting_x_list}

        return {'vv_result': vv_result, 'cal_result': self.cal_result}
    
    def predict(self, testset_spec_intersect, nlv=None, testset_indices=None, testset_target=None):
        if nlv is None:
            self.nlv = self.optimal_nlv
        else:
            self.nlv = nlv
        self.testset_spec_intersect = testset_spec_intersect
        n_test_samples = self.testset_spec_intersect.shape[0] - 1
        # --------- 根据隐变量数，生成预测所需参数 ---------
        model_parameters = self.cal_result['model_parameters']
        b = model_parameters['b'][:, self.nlv - 1]
        calx_loadings = model_parameters['calx_loadings'][:, :self.nlv]  # 保存0 ~ opt_nlv-1
        calx_scores = model_parameters['calx_scores'][:, :self.nlv]  # 保存0 ~ opt_nlv-1
        calx_scores_weights = model_parameters['calx_scores_weights'][:, :self.nlv]  # 保存0 ~ opt_nlv-1
        leverage_limit = model_parameters['leverage_limit'][self.nlv - 1]  # 保存0 ~ opt_nlv-1
        if b.ndim == 1:
            b = b[:, np.newaxis]
        if testset_indices is None:
            testset_indices = np.arange(n_test_samples)
        # ---------  测试集光谱预处理 ---------
        testspec_pretreated = self._spec_pretreat4transform(self.testset_spec_intersect)
        # --------- 根据variable indices, 截取波长点 ---------
        testspec_subset = testspec_pretreated[:, self.variable_indices]
        testx_subset = testspec_subset[1:, :]
        # --------- 开始预测 ---------
        predicte_value_temp = dot(testx_subset, b)
        predict_value = self._target_inverse_pretreat(predicte_value_temp)

        # ===================== 统计指标 =====================
        test_x_scores = dot(testx_subset, calx_scores_weights)
        # ---- Leverage & Hotelling TSquared   leverage_t2_calc(scores, x_scores)
        leverage_t2_result = leverage_t2_calc(test_x_scores, calx_scores)
        leverage = leverage_t2_result['leverage'][:, -1:]
        t2 = leverage_t2_result['t2'][:, -1:]

        if testset_target is None or n_test_samples < 2:

            return {'predict_value': predict_value,
                    'x_scores': test_x_scores,
                    't2': t2,
                    'leverage': leverage}
        else:
            # ---- 光谱残差
            test_q_result = q_calc(calx_loadings, test_x_scores, testx_subset)
            fitting_x_matrix = test_q_result['fitting_x_list'][-1]  # 提取最后1个元素
            residual_matrix = test_q_result['residual_matrix_list'][-1]  # 提取最后1个元素  (n_test_samples, n_variables)
            test_q = test_q_result['q'][:, -1:]
            test_x_residual = sqrt(test_q_result['q'][:, -1:])  # 提取最后1列
            if testset_target.ndim == 1:
                testset_target = testset_target[:, np.newaxis]

            # ---- x_fvalue and x_fprob
            x_fvalue = (n_test_samples - 1) * test_x_residual ** 2 / \
                       (sum(square(test_x_residual), axis=0) - test_x_residual ** 2)
            x_fprob = sps.distributions.f.cdf(x_fvalue, 1, n_test_samples - 1)
            # ---- leverage_limit
            x_outlier_indices = testset_indices[np.where(leverage > leverage_limit)[0]]
            # ---- 计算y_residual的fvalue和fprob
            test_y_residual = predict_value - testset_target
            y_fvalue = (n_test_samples - 1) * test_y_residual ** 2 / (
                sum(square(test_y_residual), axis=0) - test_y_residual ** 2)
            y_fprob = sps.distributions.f.cdf(y_fvalue, 1, n_test_samples - 1)
            y_outlier_indices = testset_indices[np.where(abs(y_fprob) > 0.99)[0]]

            # ---- 各种统计量
            rmse_statistics = rmse_calc(predict_value, testset_target)
            r2 = rmse_statistics['r2']
            rmsep = rmse_statistics['rmse']
            sep = rmse_statistics['sep']  # A bias corrected version of rmsep
            press = rmse_statistics['press']
            rpd = rmse_statistics['rpd']
            bias = rmse_statistics['bias']
            linear_regression_coefficient = rmse_statistics['linear_regression_coefficient']
            relative_error = rmse_statistics['relative_error']

            # ---- 20190128增加y_tvalue(学生化残差)
            prevent_invalid_for_negetive_sqrt = np.seterr(invalid='ignore')
            y_tvalue = test_y_residual / (rmsep * sqrt(1 - leverage)) 

            # 采用t检验方法确定验证集的预测值与相应的已知参考数据是否有统计意义上的偏差
            significant_difference_tvalue = np.abs(bias) * sqrt(n_test_samples) / sep
            # 95% sl=0.05 双边检验
            significant_difference_critical_value = np.array([sps.t.ppf(0.975, n_test_samples)])
            # 配对t检验 paired_test_pvalue > 0.05 则无显著性差异
            paired_ttest_statistic, paired_ttest_pvalue = sps.ttest_rel(predict_value, testset_target)

            return {'predict_value': predict_value,
                    'x_scores': test_x_scores,
                    'leverage': leverage,
                    't2': t2,
                    'q': test_q,
                    'x_residual': test_x_residual,
                    'fitting_x_list': fitting_x_matrix,
                    'residual_matrix_list': residual_matrix,
                    'x_fvalue': x_fvalue,
                    'x_fprob': x_fprob,
                    'x_outlier_indices': x_outlier_indices,
                    'y_fvalue': y_fvalue,
                    'y_fprob': y_fprob,
                    'y_tvalue': y_tvalue,  # 学生化残差
                    'y_outlier_indices': y_outlier_indices,
                    'r2': r2,
                    'rmsep': rmsep,
                    'sep': sep,
                    'press': press,
                    'rpd': rpd,
                    'bias': bias,
                    'linear_regression_coefficient': linear_regression_coefficient,
                    'relative_error': relative_error,
                    'significant_difference_tvalue': significant_difference_tvalue,
                    'significant_difference_critical_value': significant_difference_critical_value,
                    'paired_ttest_pvalue': paired_ttest_pvalue
                    }


# +++++++++++++++++++++++++++++++++++++++++++++++ Quantitative Algorithm +++++++++++++++++++++++++++++++++++++++++++++++

def ikpls_algorithm(ab, target, max_nlv):
    '''
    Improved Kernel Partial Least Squares, IKPLS
    :param ab: 光谱吸光度矩阵 (100, 700)
    :param target: (100, 1) or (100,)
    :param max_nlv:
    :return:
        b: 回归系数
        预测时：dot(ab, pls['b'][:, max_nlv-1] 得到一维数组
        预测时：dot(ab, pls['b'][:, max_nlv-1:] 得到二维数组
        x_weights: X权重矩阵 w
        x_loadings: X载荷矩阵 p
        x_scores: X得分矩阵 t
        y_loadings: y载荷向量 q
        x_scores_weights: X得分矩阵的权重矩阵 r   ---- 新样品 T = X r
    '''
    n_samples, n_variables = ab.shape
    if n_samples != target.shape[0]:
        raise ValueError('光谱数量与参考值数量不一致！')
    if max_nlv > np.min((n_samples, n_variables)):
        max_nlv = np.min((n_samples, n_variables))
    x_scores = zeros((n_samples, max_nlv))
    x_loadings = zeros((n_variables, max_nlv))
    y_loadings = zeros((1, max_nlv))
    x_weights = zeros((n_variables, max_nlv))
    x_scores_weights = zeros((n_variables, max_nlv))
    xy = dot(ab.T, target).ravel()  
    for i in range(max_nlv): # 0,1,2,3,4
        w = xy
        w = w / sqrt(dot(w.T, w))
        r = w
        for j in range(i):  # i=0时不运行
            r = r - dot(x_loadings[:, j], w) * x_scores_weights[:, j]
        t = dot(ab, r)
        tt = dot(t.T, t)
        p = dot(ab.T, t) / tt
        q = dot(r.T, xy) / tt
        xy = xy - dot(dot(p, q), tt)
        x_weights[:, i] = w
        x_loadings[:, i] = p
        x_scores[:, i] = t
        y_loadings[0, i] = q
        x_scores_weights[:, i] = r
    b = cumsum(dot(x_scores_weights, diag(y_loadings.ravel())), axis=1)
    return {'b': b,
            'x_scores': x_scores,
            'x_loadings': x_loadings,
            'y_loadings': y_loadings,
            'x_scores_weights': x_scores_weights,
            'x_weights': x_weights,
            'max_nlv':max_nlv}

def nipals_algorithm(ab, target, max_nlv):  # ab(700,700) calset_target(700,1)or(700,)  max_nlv(15)
    '''
    Nonlinear Iterative Partial Least Squares，NIPALS
    :param ab:
    :param target:
    :param max_nlv:
    :return:
    '''
    n_samples, n_variables = ab.shape
    if n_samples != target.shape[0]:
        raise ValueError('光谱数量与参考值数量不一致！')
    if max_nlv > np.min((n_samples, n_variables)):
        max_nlv = np.min((n_samples, n_variables))
    x_scores = zeros((n_samples, max_nlv))    # (700,15)
    x_loadings = zeros((n_variables, max_nlv))  # (700,15)
    y_loadings = zeros((1, max_nlv))                # (1,15)
    x_weights = zeros((n_variables, max_nlv))   #(700,15)
    for i in range(max_nlv):
        xy = dot(ab.T, target).ravel()  
        x_weights[:, i] = xy / norm(xy)
        x_scores[:, i] = dot(ab, x_weights[:, i])
        x_loadings[:, i] = dot(ab.T, x_scores[:, i]) / dot(x_scores[:, i].T, x_scores[:, i])
        y_loadings[0, i] = dot(x_scores[:, i].T, target) / dot(x_scores[:, i].T, x_scores[:, i])
        ab = ab - outer(x_scores[:, i], x_loadings[:, i])   #外积，得到矩阵

    x_scores_weights = dot(x_weights, inv(dot(x_loadings.T, x_weights)))
    b = cumsum(dot(x_scores_weights, diag(y_loadings.ravel())), axis=1)  #y_loadings拉成一维数组

    return {'b': b, 'x_scores': x_scores, 'x_loadings': x_loadings, 'y_loadings': y_loadings, \
            'x_scores_weights': x_scores_weights, 'x_weights': x_weights}

def simpls_algorithm(ab, target, max_nlv):
    '''
    Straightforward Implementation of a statistically inspired Modification of the Partial Least Squares, SIMPLS
    :param ab:
    :param target:
    :param max_nlv:
    :return:
    '''
    n_samples, n_variables = ab.shape
    if np.ndim(target) == 1:
        target = target[:, np.newaxis]
    if n_samples != target.shape[0]:
        raise ValueError('光谱数量与参考值数量不一致！')
    if max_nlv > np.min((n_samples, n_variables)):
        max_nlv = np.min((n_samples, n_variables))
    V = zeros((n_variables, max_nlv))
    x_scores = zeros((n_samples, max_nlv))  # X scores (standardized)
    x_weights = zeros((n_variables, max_nlv))  # X weights
    x_loadings = zeros((n_variables, max_nlv))  # X loadings
    y_loadings = zeros((1, max_nlv))  # Y loadings
    y_scores = zeros((n_samples, max_nlv))  # Y scores
    s = dot(ab.T, target).ravel()  # cross-product matrix between the ab and target_data
    for i in range(max_nlv):
        r = s
        t = dot(ab, r)
        tt = norm(t)
        t = t / tt
        r = r / tt
        p = dot(ab.T, t)
        q = dot(target.T, t)
        u = dot(target, q)
        v = p  # P的正交基
        if i > 0:
            v = v - dot(V, dot(V.T, p))  # Gram-Schimidt orthogonal
            u = u - dot(x_scores, dot(x_scores.T, u))
        v = v / norm(v)
        s = s - dot(v, dot(v.T, s))
        x_weights[:, i] = r
        x_scores[:, i] = t
        x_loadings[:, i] = p
        y_loadings[:, i] = q
        y_scores[:, i] = u
        V[:, i] = v
    b = cumsum(dot(x_weights, diag(y_loadings.ravel())), axis=1)

    return {'b': b, 'x_scores': x_scores, 'x_loadings': x_loadings, 'y_loadings': y_loadings, \
            'x_scores_weights': x_weights, 'x_weights': x_weights, 'y_scores':y_scores}


# +++++++++++++++++++++++++++++++++++++++++++++++ Sampling Algorithm +++++++++++++++++++++++++++++++++++++++++++++++
# Note: All the sampling indices should be sorted, using np.sort()

def cv_kfold_random_sampling(n_population, kfold=9, seed=999):
    '''
    The first ``n % kfold`` folds have size ``n // kfold + 1``,
    other folds have size ``n // kfold``, where ``n`` is the number of samples.
    '''
    train_indices_list = []
    test_indices_list = []
    rng = np.random.RandomState(seed)
    fold_sizes = (n_population // kfold) * np.ones(kfold, dtype=np.int)  # other folds
    fold_sizes[:n_population % kfold] += 1  # background 'n % kfold' folds
    population_indices = np.arange(n_population)  # 83
    for i in range(kfold):
        mask = zeros(n_population, dtype=np.bool)
        test_indices = rng.choice(population_indices, fold_sizes[i], replace=False)
        mask[test_indices] = True  # 选中的标记True
        train_indices = np.arange(n_population)[~mask]  # 用于训练集
        test_indices_list.append(test_indices)
        train_indices_list.append(train_indices)

    return train_indices_list, test_indices_list

def cv_kfold_systematic_sampling(n_population, kfold=9):
    '''
    前面的mod折数量多, 后面的折数量少
    The first ``n % kfold`` folds have size ``n // kfold + 1``,
    other folds have size ``n // kfold``, where ``n`` is the number of samples.
    '''
    train_indices_list = []
    test_indices_list = []

    fold_sizes = ((n_population // kfold) + 1 ) * np.ones(kfold, dtype=np.int)
    fold_sizes[n_population % kfold:] -= 1

    for i in range(kfold):
        mask = zeros(n_population, dtype=np.bool)
        test_indices = np.linspace(start=i, stop=kfold*(fold_sizes[i] - 1) + i, num=fold_sizes[i], dtype=int)
        test_indices_list.append(test_indices)
        mask[test_indices] = True  
        train_indices = np.arange(n_population)[~mask]
        train_indices_list.append(train_indices)

    return train_indices_list, test_indices_list

def cv_lpo_random_sampling(n_population, p=3, seed=999):
    '''
    保证前面的 n // p 折数量是p, 最后一折数量是 n % p
    The last  fold have size ``n % p``, other folds have size ``p``
    '''
    kfold = (n_population + p -1) // p  # 向上取整 等价于 math.ceil(n_population / p)
    mod = n_population % p
    if mod == 0:
        kfold = n_population // p
        fold_sizes = p * np.ones(kfold, dtype=np.int)
    else:
        kfold = n_population // p + 1
        fold_sizes = p * np.ones(kfold, dtype=np.int)
        fold_sizes[-1] = mod
    train_indices_list = []
    test_indices_list = []
    rng = np.random.RandomState(seed)
    population_indices = np.arange(n_population)  # 83
    temp_left_indices = population_indices
    for i in range(kfold):
        test_indices = rng.choice(temp_left_indices, fold_sizes[i], replace=False)
        train_indices = np.setdiff1d(population_indices, test_indices)  # 用于训练集
        test_indices_list.append(test_indices)
        train_indices_list.append(train_indices)
        temp_left_indices = np.setdiff1d(temp_left_indices, test_indices)

    return train_indices_list, test_indices_list

def cv_lpo_systematic_sampling(n_population, p=3):
    '''
    Leave p Out:系统采样作为取出P个样本的依据; 内部交叉验证从第一个开始取出作为验证集(0, 5, 10 ... )
    :param n_population:
    :param p:
    :return:
    '''
    population_indices = np.arange(n_population, dtype=int)
    train_indices_list = []
    test_indices_list = []
    mod = n_population % p
    # ================ 能整除 ================
    if mod == 0:  # 能整除
        kfold = n_population // p
        interval = n_population // p
        fold_sizes = p * np.ones(kfold, dtype=np.int)
        for i in range(kfold):
            test_indices = np.linspace(start=i, stop=(fold_sizes[i] - 1) * interval + i, num=fold_sizes[i], dtype=int)
            test_indices_list.append(test_indices)
            train_indices = np.setdiff1d(population_indices, test_indices)
            train_indices_list.append(train_indices)
    # ================ 不能整除 ================
    else:  # 不能整除
        kfold = n_population // p + 1
        interval = n_population // p
        fold_sizes = p * np.ones(kfold, dtype=np.int)
        fold_sizes[-1] = mod  # 最后一折个数为余数  14 % 4 = 2
        # ---------------- 处理前(kfold - 1)折 ----------------
        for i in range(kfold - 1):
            test_indices = np.linspace(start=i, stop=(fold_sizes[i] - 1) * interval + i, num=fold_sizes[i], dtype=int)
            test_indices_list.append(test_indices)
            train_indices = np.setdiff1d(population_indices, test_indices)
            train_indices_list.append(train_indices)
        # ---------------- 处理最后一折 ----------------
        last_fold_test_indices = population_indices[-mod:]
        last_fold_train_indices = np.setdiff1d(population_indices, last_fold_test_indices)
        # ---------------- 合并最后一折 ----------------
        test_indices_list.append(last_fold_test_indices)
        train_indices_list.append(last_fold_train_indices)

    return train_indices_list, test_indices_list

def montecarlo_sampling(n_population, test_size=0.2, seed=999):
    '''
    用于蒙特卡洛采样，原理是随机采样
    :param n_population:
    :param test_size:
    :param seed:
    :return:
    '''
    rng = np.random.RandomState(seed)
    n_test = int(n_population * test_size)  # 81*0.2=16
    # n_train = n - n_test
    mask = zeros(n_population, dtype=np.bool)
    a = np.arange(n_population)
    rng.shuffle(a)
    test_indices = rng.choice(a, n_test, replace=False)
    test_indices.sort()
    mask[test_indices] = True
    train_indices = np.arange(n_population)[~mask]
    # ================ 重新排序 ================
    train_indices.sort()
    test_indices.sort()

    return train_indices, test_indices

def ks_sampling(X, p=3, population_indices=None):
    '''
    用于ks抽样，返回指定样品数目的索引号(因为依次取出样品的顺序有内在特性，不再排序)
    :param X: 针对吸光度矩阵
    :param p:
    :param population_indices:
    :return:
    '''
    n_samples, n_variables = X.shape
    if population_indices is None:
        population_indices = np.arange(n_samples)
    D = squareform(pdist(X, metric='euclidean'))
    temp_index = []
    index_2max = where(D == D.max())[0]
    temp_index.append(index_2max[0])
    temp_index.append(index_2max[1])
    retained_D = D[:, temp_index]
    retained_D[temp_index, :] = 0
    for k in range(p - 2):
        choice_index = where(retained_D == max(min(retained_D, axis=1, keepdims=True)))[0][0]
        temp_index.append(choice_index)
        retained_D = D[:, temp_index]
        retained_D[temp_index, :] = 0

    train_indices = np.sort(population_indices[temp_index])
    test_indices = np.setdiff1d(population_indices, train_indices)

    return train_indices, test_indices

def spxy_sampling(X, y, p=3, population_indices=None):
    '''
    用于SPXY抽样, 返回指定样品数目的索引号(因为依次取出样品的顺序有内在特性，不再排序)
    :param X:
    :param y:
    :param p:
    :param population_indices:
    :return:
    '''
    n_samples, n_variables = X.shape
    if y.ndim == 1:
        y = y[:, np.newaxis]
    if population_indices is None:
        population_indices = np.arange(n_samples)
    # ----------- 光谱距离计算 -----------
    D_ab = zeros((n_samples, n_samples))
    for i in range(n_samples - 1):  
        for j in range(i+1, n_samples):
            D_ab[i, j] = norm(X[i, :]-X[j, :])
    D_ab += D_ab.T 
    D_ab_max = np.max(D_ab)
    # ----------- 浓度距离计算 -----------
    D_con = zeros((n_samples, n_samples))
    for i in range(n_samples - 1): 
        for j in range(i+1, n_samples):
            D_con[i, j] = norm(y[i, :]-y[j, :])
    D_con += D_con.T  
    D_con_max = np.max(D_con)

    # ----------- 光谱&浓度距离 -----------
    D = D_ab / D_ab_max + D_con / D_con_max

    # ----------- 抽样 -----------
    temp_index = []
    index_2max = where(D == D.max())[0]
    temp_index.append(index_2max[0])
    temp_index.append(index_2max[1])
    retained_D = D[:, temp_index]
    retained_D[temp_index, :] = 0
    for k in range(p - 2):
        choice_index = where(retained_D == max(min(retained_D, axis=1, keepdims=True)))[0][0]
        temp_index.append(choice_index)
        retained_D = D[:, temp_index]
        retained_D[temp_index, :] = 0

    cal_indices = population_indices[temp_index]
    val_indices = np.setdiff1d(population_indices, cal_indices)

    return cal_indices, val_indices

def samples_systematic_split(X, val_size=0.1, test_size=0, population_indices=None):
    '''
    系统采样原理
    :param X: Absorbance
    :param n_population:
    :param val_size:
    :param test_size:
    :return:
    '''
    n_population = X.shape[0]
    if (val_size + test_size) >= 1.0:
        raise ValueError('Wrong parameters of the sampling ratio!')
    n_val = int(n_population * val_size)
    n_test = int(n_population * test_size)
    n_train = n_population - n_val - n_test
    n_val_test = n_val + n_test
    if population_indices is None:
        population_indices = np.arange(n_population)

    # -------------- 先挑选 val_test_set，同分布 --------------
    interval_1 = n_population // n_val_test  # 有

    if interval_1 > 1: 
        val_test_indices = np.array([population_indices[interval_1 * i - 1] for i in range(1, n_val_test+1)])

    elif interval_1 == 1: # 不够等距采样
        val_test_indices_first = np.array([population_indices[2 * i - 1] for i in range(1, n_population//2 + 1)])
        val_test_indices_last = np.array([population_indices[2 * i] for i in range(1, n_val_test - n_population//2 + 1)])
        val_test_indices = np.hstack((val_test_indices_first, val_test_indices_last))

    train_indices = np.setdiff1d(population_indices, val_test_indices)

    if n_test == 0:
        val_indices = val_test_indices
        test_indices = np.setdiff1d(val_test_indices, val_indices)
    elif n_val == 0:
        test_indices = val_test_indices
        val_indices = np.setdiff1d(val_test_indices, test_indices)
    # -------------- 再从val_test_set中挑选valset，同分布 --------------
    else:
        interval_2 = n_val_test // n_val  # 先挑选出 valset
        if interval_2 > 1:  
            val_indices = np.array([val_test_indices[interval_2 * j - 1] for j in range(1, n_val + 1)])
        elif interval_2 == 1:  
            val_indices_first = np.array([val_test_indices[2 * j - 1] for j in range(1, n_val_test//2 + 1)])
            val_indices_last = np.array([val_test_indices[2 * j] for j in range(1, n_val - n_val_test // 2 + 1)])
            val_indices = np.hstack((val_indices_first, val_indices_last))
        test_indices = np.setdiff1d(val_test_indices, val_indices)

    # ================ 重新排序 ================
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    return train_indices, val_indices, test_indices

def samples_ks_split(X, val_size=0.1, test_size=0.1, population_indices=None):
    '''
    样品划分，Kennard-Stone原理
    :param X: Absorbance
    :param val_size:
    :param test_size:
    :param population_indices:
    :return:
    '''
    n_population = X.shape[0]
    if (val_size + test_size) >= 1.0:
        raise ValueError('Wrong parameters of the sampling ratio!')
    n_val = int(n_population * val_size)
    n_test = int(n_population * test_size)
    n_train = n_population - n_val - n_test
    if population_indices is None:
        population_indices = np.arange(n_population)

    # 先挑选trainset
    train_indices, val_test_indices = ks_sampling(X, n_train, population_indices=population_indices)
    val_test_set = X[val_test_indices, :]

    if n_test == 0:
        val_indices = val_test_indices
        test_indices = np.setdiff1d(val_test_indices, val_indices)
    elif n_val == 0:
        test_indices = val_test_indices
        val_indices = np.setdiff1d(val_test_indices, test_indices)
    else:
        # 再挑选valset
        val_indices, test_indices = ks_sampling(val_test_set, n_val, population_indices=val_test_indices)

    # ================ 重新排序 ================
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    return train_indices, val_indices, test_indices

def samples_spxy_split(X, target, val_size=0.1, test_size=0.1, population_indices=None):
    '''
    样品划分，基于指定的target
    :param X: Absorbance
    :param target:
    :param val_size:
    :param test_size:
    :param population_indices:
    :return:
    '''
    if target.ndim == 1:
        target = target[:, np.newaxis]
    n_population = X.shape[0]
    if (val_size + test_size) >= 1.0:
        raise ValueError('Wrong parameters of the sampling ratio!')
    n_val = int(n_population * val_size)
    n_test = int(n_population * test_size)
    n_train = n_population - n_val - n_test
    if population_indices is None:
        population_indices = np.arange(n_population)

    # 先挑选trainset
    train_indices, val_test_indices = spxy_sampling(X, target,
                                                    n_train, population_indices=population_indices)
    val_test_set_ab = X[val_test_indices, :]
    val_test_set_con = target[val_test_indices, :]

    if n_test == 0:
        val_indices = val_test_indices
        test_indices = np.setdiff1d(val_test_indices, val_indices)
    elif n_val == 0:
        test_indices = val_test_indices
        val_indices = np.setdiff1d(val_test_indices, test_indices)
    else:
        # 再挑选valset
        val_indices, test_indices = spxy_sampling(val_test_set_ab, val_test_set_con,
                                                  n_val, population_indices=val_test_indices)

    # ================ 重新排序 ================
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    return train_indices, val_indices, test_indices

def samples_random_split(X, val_size=0.1, test_size=0.1, seed=999, population_indices=None):
    '''

    :param X: Absorbance
    :param val_size:
    :param test_size:
    :param seed:
    :param population_indices:
    :return:
    '''
    rng = np.random.RandomState(seed)
    n_population = X.shape[0]
    if (val_size + test_size) >= 1.0:
        raise ValueError('Wrong parameters of the sampling ratio!')
    n_val = int(n_population * val_size)
    n_test = int(n_population * test_size)
    n_train = n_population - n_val - n_test
    if population_indices is None:
        population_indices = np.arange(n_population)
    train_indices = population_indices[rng.choice(n_population, n_train, replace=False)]
    val_test_indices = np.setdiff1d(population_indices, train_indices)
    val_indices = val_test_indices[rng.choice(n_val + n_test, n_val, replace=False)]
    test_indices = np.setdiff1d(val_test_indices, val_indices)

    # ================ 重新排序 ================
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    return train_indices, val_indices, test_indices


# +++++++++++++++++++++++++++++++++++++++++++++++ Utilities +++++++++++++++++++++++++++++++++++++++++++++++

class PLSR(object):
    '''
    用于 PLS Regression，指定PLS算法、最大潜变量数
    必要输入参数包括：吸光度矩阵、参考值
    可以实现：校正、预测(所有光谱必须事先兼容)
    '''

    def __init__(self, algorithm='ikpls_algorithm', max_nlv=10):
        self.algorithm = algorithm
        self.max_nlv = max_nlv

        return

    def fit(self, cal_spec, cal_target):
        '''
        PLS 回归得到 b, x_scores, x_loadings, y_loadings, x_scores_weights, x_weights, max_nlv
        :param cal_ab: 使用pretreat.ConstructCompatiblePLSBand().fit_construct生成
        :param cal_target:
        :return:
        '''
        if cal_target.ndim == 1:
            cal_target = cal_target[:, np.newaxis]  # 用于多维结果的broadcast计算
        cal_ab = cal_spec[1:, :]
        # --------- 处理维数过大的问题 ---------
        if self.max_nlv > np.min((cal_ab.shape[0], cal_ab.shape[1])):
            self.max_nlv = np.min((cal_ab.shape[0], cal_ab.shape[1]))
        # --------- PLS回归 ---------
        self.pls_result = eval(self.algorithm)(cal_ab, cal_target, self.max_nlv)

        return self

    def fit_predict(self, cal_spec, cal_target):
        cal_ab = cal_spec[1:, :]
        self.model_ab = cal_ab
        if cal_target.ndim == 1:
            cal_target = cal_target[:, np.newaxis]  # 用于多维结果的broadcast计算
        self.fit(cal_spec, cal_target)
        # pls_result由ikpls_algorithm算法给出以下数据：
        # b, x_scores, x_loadings, y_loadings, x_scores_weights, x_weights, max_nlv
        b = self.pls_result['b']
        calx_scores = self.pls_result['x_scores']
        calx_loadings = self.pls_result['x_loadings']
        calx_scores_weights = self.pls_result['x_scores_weights']
        fit_value = dot(cal_ab, b)  # 全部维的结果

        pls_calibration_model = {'b':b,
                                 'fit_value': fit_value,
                                 'algorithm': self.algorithm,
                                 'max_nlv': self.max_nlv,
                                 'pls_result': self.pls_result,
                                 'x_loadings': calx_loadings,
                                 'x_scores': calx_scores,
                                 'x_scores_weights':calx_scores_weights,
                                 'model_ab': self.model_ab}

        return {'b':b,
                'fit_value': fit_value,
                'algorithm': self.algorithm,
                'max_nlv': self.max_nlv,
                'pls_result': self.pls_result,
                'x_loadings': calx_loadings,
                'x_scores_weights': calx_scores_weights,
                'x_scores': calx_scores,
                'model_ab': self.model_ab,
                'pls_calibration_model': pls_calibration_model}

    def val_predict(self, val_spec):
        '''
        调用当前实例中校正集校正的结果
        既可以用于不含浓度的样品的预测，也可以用于验证集验证
        :param val_ab: numpy.ndarray
        :return:
        '''
        val_ab = val_spec[1:, :]
        b = self.pls_result['b']
        # 计算
        predict_value = dot(val_ab, b)

        return {'predict_value':predict_value}

def q_calc(calx_loadings, scores, pretreated_data):
    '''
    Usually the statistic Q, also called squared prediction error(SPE),
    and the Hotelling's T^2 statistic are used to represent the variability in the residual subspace
    and principal component subspace
    For PCAValidation: pretreated_data
    For PLS: pretreated_ab
    :param calx_loadings:  校正模型的 x_loadings
    :param scores:  待计算样本的 scores
    :param pretreated_data:  待计算样本处理之后的进入pca/pls算法的数据
    :return: q、残差矩阵列表、拟合光谱矩阵列表
    '''
    if calx_loadings.ndim == 1:
        calx_loadings = calx_loadings[:, np.newaxis]  # 如果一维数组，增加至二维数组
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]  # 如果一维数组，增加至二维数组
    if pretreated_data.ndim == 1:
        pretreated_data = pretreated_data[:, np.newaxis]  # 如果一维数组，增加至二维数组
    n_samples, n_lv = scores.shape
    n_variables = calx_loadings.shape[0]
    q = zeros((n_samples, n_lv))  # Sometimes referred to as the Squared Prediction Error (SPE)
    f_residuals = zeros((n_samples, n_lv))
    residual_matrix_list = []
    fitting_x_list = []
    x_variable_residuals = zeros((n_variables, n_lv))  # (n_variables, n_lv)
    x_sample_residuals = zeros((n_samples, n_lv))
    for i in range(n_lv):  # 0:5 nlv
        # Q
        fitting_x_lv = dot(scores[:, :i + 1], calx_loadings[:, :i + 1].T)
        residual_matrix_lv = pretreated_data - fitting_x_lv
        residual_matrix_list.append(residual_matrix_lv)
        fitting_x_list.append(fitting_x_lv)
        q_lv = np.sum(residual_matrix_lv ** 2, axis=1)
        f_residuals_lv = sqrt(np.mean(residual_matrix_lv ** 2, axis=1))
        q[:, i] = q_lv
        f_residuals[:, i] = f_residuals_lv
        x_sample_residuals[:, i] = np.sum(residual_matrix_lv ** 2, axis=1) / n_variables
        x_variable_residuals[:, i] = np.sum(residual_matrix_lv ** 2, axis=0) / n_samples
    x_total_residuals = np.mean(x_variable_residuals, axis=0, keepdims=True)  # (1, n_lv)
    explained_x_sample_variance = (1 - x_sample_residuals / (np.sum(pretreated_data ** 2, axis=1, keepdims=True) / \
                                                             n_variables)) * 100
    explained_x_variable_variance = (1-x_variable_residuals.T / (np.sum(pretreated_data ** 2, axis=0)/n_samples)) * 100
    explained_x_total_variance = (1 - x_total_residuals / np.mean(pretreated_data ** 2)) * 100
    explained_x_variance_ratio = np.hstack((explained_x_total_variance[:, 0:1], np.diff(explained_x_total_variance)))

    return {'q':q,
            'f_residuals': f_residuals,
            'residual_matrix_list':residual_matrix_list,
            'fitting_x_list':fitting_x_list,
            'x_sample_residuals': x_sample_residuals,
            'x_variable_residuals':x_variable_residuals,
            'x_total_residuals': x_total_residuals,
            'explained_x_sample_variance': explained_x_sample_variance,
            'explained_x_variable_variance': explained_x_variable_variance.T,
            'explained_x_total_variance': explained_x_total_variance,
            'explained_x_variance_ratio': explained_x_variance_ratio
            }

def q_calc_cv(calx_loadings, scores, pretreated_data):
    '''
    Usually the statistic Q, also called squared prediction error(SPE),
    and the Hotelling's T^2 statistic are used to represent the variability in the residual subspace
    and principal component subspace
    For PCAValidation: pretreated_data
    For PLS: pretreated_ab
    :param calx_loadings:  校正模型的 x_loadings
    :param scores:  待计算样本的 scores
    :param pretreated_data:  待计算样本处理之后的进入pca/pls算法的数据
    :return: q、残差矩阵列表、拟合光谱矩阵列表
    '''
    if calx_loadings.ndim == 1:
        calx_loadings = calx_loadings[:, np.newaxis]  # 如果一维数组，增加至二维数组
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]  # 如果一维数组，增加至二维数组
    if pretreated_data.ndim == 1:
        pretreated_data = pretreated_data[:, np.newaxis]  # 如果一维数组，增加至二维数组
    n_samples, n_lv = scores.shape
    # n_variables = calx_loadings.shape[0]
    q = zeros((n_samples, n_lv))  # Sometimes referred to as the Squared Prediction Error (SPE)
    residual_matrix_list = []
    fitting_x_list = []
    for i in range(n_lv):  # 0:5 nlv
        # Q
        fitting_x_lv = dot(scores[:, :i + 1], calx_loadings[:, :i + 1].T)
        residual_matrix_lv = pretreated_data - fitting_x_lv
        residual_matrix_list.append(residual_matrix_lv)
        fitting_x_list.append(fitting_x_lv)
        q_lv = np.sum(residual_matrix_lv ** 2, axis=1)
        q[:, i] = q_lv

    return {'q':q,
            'residual_matrix_list':residual_matrix_list,
            'fitting_x_list':fitting_x_list}

def leverage_t2_calc(scores, calx_scores):
    '''
    Leverage & Hotelling T2
    :param scores: 待计算样本的 scores
    :param calx_scores:  校正模型样本的 x_scores
    :return: t2
    '''
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]  # 如果一维数组，增加至二维数组
    if calx_scores.ndim == 1:
        calx_scores = calx_scores[:, np.newaxis]  # 如果一维数组，增加至二维数组
    n_cal_samples = calx_scores.shape[0]
    leverage = zeros((scores.shape[0], scores.shape[1]))
    for i in range(scores.shape[1]):  # 0:5 nlv
        lev_lv = diag(dot(dot(scores[:, :i + 1], inv(dot(calx_scores[:, :i + 1].T, calx_scores[:, :i + 1]))),
                            scores[:, :i + 1].T)) + 1 / n_cal_samples
        leverage[:, i] = lev_lv
    t2 = (n_cal_samples - 1) * (leverage - 1 / n_cal_samples)

    return {'leverage': leverage, 't2': t2}

def leverage_t2_calc_cv(cv_x_scores, calx_scores):
    '''
    Leverage & Hotelling T2 For Cross Validation
    :param cv_x_scores: 验证完成后全体样本的 scores
    :param calx_scores:  全体校正样本的 x_scores
    :return: t2
    '''
    if cv_x_scores.ndim == 1:
        cv_x_scores = cv_x_scores[:, np.newaxis]  # 如果一维数组，增加至二维数组
    if calx_scores.ndim == 1:
        calx_scores = calx_scores[:, np.newaxis]  # 如果一维数组，增加至二维数组
    n_cal_samples = calx_scores.shape[0]
    leverage = zeros((cv_x_scores.shape[0], cv_x_scores.shape[1]))
    for i in range(cv_x_scores.shape[1]):  # 0:5 nlv
        lev_lv = diag(dot(dot(cv_x_scores[:, :i + 1], inv(dot(calx_scores[:, :i + 1].T, calx_scores[:, :i + 1]))),
                            cv_x_scores[:, :i + 1].T)) + 1 / n_cal_samples
        leverage[:, i] = lev_lv
    t2 = (n_cal_samples - 1) * leverage

    return {'leverage': leverage, 't2': t2}

def outlier_detect(sample_leverage, leverage_limit, y_fprob, sample_indices=None):
    '''
    X-direction: > 3 * mean(cal_leverage)
    y-direction: F distribution fprob > 0.99
    :param sample_leverage: 二维  61 * 20 nlv
    :param leverage_limit:  1维  (20,)
    :param
    :return:
    '''
    if sample_leverage.ndim == 1:
        sample_leverage = sample_leverage[:, np.newaxis]
    if y_fprob.ndim == 1:
        y_fprob = y_fprob[:, np.newaxis]

    if sample_indices is None:
        temp_indices = np.arange(sample_leverage.shape[0])
    else:
        temp_indices = sample_indices

    # X direction: > 3 * mean(leverage, axis=0)
    x_outlier_indices_list = []  # 有10个潜变量数
    leverage_limit = leverage_limit  # (max_ncomp,)
    for i in range(sample_leverage.shape[1]):  # 0 : max_ncomp-1
        x_outlier_indices = temp_indices[np.where(sample_leverage[:, i] > leverage_limit[i])[0]]
        x_outlier_indices_list.append(x_outlier_indices)

    # y direction:
    y_outlier_indices_list = []
    for i in range(sample_leverage.shape[1]):  # 不同潜变量数
        y_outlier_indices = temp_indices[np.where(abs(y_fprob[:, i].ravel()) > 0.99)[0]]
        y_outlier_indices_list.append(y_outlier_indices)

    just_x_outlier_list = []
    just_y_outlier_list = []
    both_xy_outlier_list = []
    for k in range(sample_leverage.shape[1]):
        x_idx = x_outlier_indices_list[k]
        y_idx = y_outlier_indices_list[k]
        both_xy_outlier = np.intersect1d(x_idx, y_idx)
        just_x_outlier = np.setdiff1d(x_idx, both_xy_outlier)
        just_y_outlier = np.setdiff1d(y_idx, both_xy_outlier)

        just_x_outlier_list.append(just_x_outlier)
        just_y_outlier_list.append(just_y_outlier)
        both_xy_outlier_list.append(both_xy_outlier)

    return {'x_outlier_indices_list': x_outlier_indices_list,
            'y_outlier_indices_list': y_outlier_indices_list,
            'just_x_outlier_list': just_x_outlier_list,
            'just_y_outlier_list': just_y_outlier_list,
            'both_xy_outlier_list': both_xy_outlier_list
            }

def rmse_calc(predict_value, reference_value):
    '''
    只能用于内部交叉验证和预测
    :param predict_value:
    :param reference_value:
    :return:
    '''
    if predict_value.ndim == 1:
        predict_value = predict_value[:, np.newaxis]  # 如果一维数组，增加至二维数组
    if reference_value.ndim == 1:
        reference_value = reference_value[:, np.newaxis]  # 如果一维数组，增加至二维数组
    max_nlv = predict_value.shape[1]
    n_samples = reference_value.shape[0]
    error = predict_value - reference_value
    press = np.sum(error * error, axis=0)  # Error Sum of Squares(SSE)
    rmse = sqrt(press / n_samples)
    sst = np.sum((reference_value - mean(reference_value)) ** 2)  # Total Sum Of Squares(SST) 总离差平方和
    ssr = np.sum((predict_value - mean(reference_value)) ** 2, axis=0)  # Regression Sum of Squares(SSR)
    r2 = 1 - press / sst
    sd = sqrt(sst / (n_samples - 1))
    bias = np.mean(error, axis=0)  # 也可以叫验证平均误差

    # refer to OPUS
    # SEP (Standard Error of Prediction)
    SEP = sqrt((np.sum((error - bias) * (error - bias), axis=0)) / (n_samples - 1))
    rpd = sd / SEP

    # # correlation coefficient
    # fit_value_mc = predict_value - mean(predict_value, axis=0)
    # reference_value_mc = reference_value - mean(reference_value, axis=0)
    # corr_coeff_numerator = np.sum(fit_value_mc * reference_value_mc, axis=0)
    # corr_coeff_denominator = sqrt(np.sum(fit_value_mc ** 2, axis=0) * np.sum(reference_value_mc ** 2, axis=0))
    # correlation_coefficient = corr_coeff_numerator / corr_coeff_denominator

    # 数据线性回归(横坐标reference_value, 纵坐标predict_value)
    # linear_regression_coefficient (2, max_nlv) slope,intercept
    linear_regression_coefficient = zeros((2, max_nlv))
    for i in range(max_nlv):
        reg_coeff = lsr(reference_value, predict_value[:, i], order=1)['regression_coefficient']
        linear_regression_coefficient[:, i] = reg_coeff.ravel()

    relative_error = np.abs(error) / reference_value

    return {'r2': r2, 'rmse': rmse, 'sep':SEP, 'press': press, 'rpd': rpd, 'bias': bias,
            'linear_regression_coefficient':linear_regression_coefficient,
            'relative_error':relative_error}

def verify_customized_regions(intersect_wavelength, customized_regions, threshold=10):
    # +++++++++++++++++++++++ sub_function: used for comparing two list +++++++++++++++++++++++
    def _check_region(list1, list2, threshold=10):
        list = [list1, list2]
        list_sort = sorted([sorted(region) for region in list])
        forward_first = list_sort[0][0]
        forward_last = list_sort[0][1]
        backward_first = list_sort[1][0]
        backward_last = list_sort[1][1]
        if (backward_last - forward_last) <= 0:
            new_list = [list_sort[0]]
        elif (backward_first - forward_last) <= 0 or 0 < (backward_first - forward_last) <= threshold:
            new_list = [[forward_first, backward_last]]
        elif (backward_first - forward_last) > threshold:
            new_list = list_sort
        return new_list

    wavelength_start2end = [intersect_wavelength[0], intersect_wavelength[-1]]
    # +++++++++++++++++++++++ merge regions +++++++++++++++++++++++
    # step1. sort every region, e.g. [10000, 7000] -> [7000, 10000]
    # step2. sort the list, e.g. [[4000, 5000], [7000, 10000], [6000, 8000]] -> [[4000, 5000], [6000, 8000], [7000, 10000]]
    region_list_sort = sorted([sorted(region) for region in customized_regions])
    n_regions = len(region_list_sort)
    merged_list = []
    temp_list = [region_list_sort[0]]
    if n_regions == 1:
        merged_list = region_list_sort
    elif n_regions > 1:
        for i in range(n_regions - 1):
            list1 = temp_list[-1]
            list2 = region_list_sort[i + 1]  # 从1开始取出
            temp_list = _check_region(list1, list2, threshold=threshold)
            if len(temp_list) == 2:
                merged_list.append(temp_list[0])
        merged_list.append(temp_list[-1])
    # +++++++++++++++++++++++ validate the merged_list +++++++++++++++++++++++
    n_merged_regions = len(merged_list)
    valid_region_sort = sorted(wavelength_start2end)
    valid_start = valid_region_sort[0]
    valid_end = valid_region_sort[1]
    verified_regions = []
    for j in range(n_merged_regions):
        pending_list = merged_list[j]
        pending_start = pending_list[0]
        pending_end = pending_list[1]
        if pending_start > valid_end or pending_end < valid_start:
            temp_valid = [valid_start, valid_end]
        elif pending_start < valid_start and valid_start < pending_end < valid_end:
            temp_valid = [valid_start, pending_end]
        elif valid_start < pending_start < valid_end and pending_end > valid_end:
            temp_valid = [pending_start, valid_end]
        elif pending_start > valid_start and pending_end < valid_end:
            temp_valid = pending_list
        elif pending_start <= valid_start and pending_end >= valid_end:
            temp_valid = [valid_start, valid_end]
        else:
            temp_valid = [pending_start, pending_end]
        verified_regions.append(temp_valid)
    if len(verified_regions) == 0:
        raise ValueError('选择的谱区与有效谱区不匹配，请重新选择！')
    else:
        return verified_regions

def generate_variable_indices(intersect_wavelength, customized_regions, threshold=10):
    '''
    根据各个兼容光谱形成的有效谱区(波长交集)，来判断自定义的谱区列表，合并重合部分，舍弃多余部分
    :param intersect_wavelength
    :param customized_regions: 二层嵌套列表, like [[4000, 5000], [10000, 7000], [6000, 8000]]
    :param threshold: cm-1 / nm
    :return:
    '''
    verified_regions = verify_customized_regions(intersect_wavelength, customized_regions, threshold=threshold)
    n_valid_regions = len(verified_regions)
    indices_list = []
    for i in range(n_valid_regions):
        valid_region = verified_regions[i]
        valid_region_start = valid_region[0]
        valid_region_end = valid_region[1]
        if intersect_wavelength[0] > valid_region_end or intersect_wavelength[-1] < valid_region_start:
            continue
        else:
            start_index = np.argmin(np.abs(intersect_wavelength - valid_region_start))
            end_index = np.argmin(np.abs(intersect_wavelength - valid_region_end))

            if valid_region_start < intersect_wavelength[start_index]:
                start_index -= 1
            if valid_region_end > intersect_wavelength[end_index]:
                end_index += 1

            indices = np.array(np.arange(start_index, end_index + 1))
            indices_list.append(indices)
    variable_indices = np.hstack([ind for ind in indices_list])  # 20200321修改成[]

    return variable_indices


# +++++++++++++++++++++++++++++++++++++++++++++++ Pretreat Class +++++++++++++++++++++++++++++++++++++++++++++++

# ================ Class 用于光谱矩阵操作、建模、预测过程中转换 ================

# -------- 多样本操作 --------

class MC(object):
    '''
    Mean Centering 均值中心化
    '''
    def __init__(self, avg_ab=None):
        self.avg_ab = avg_ab
        return

    def mc(self, spec, avg_ab=None):
        wavelength = spec[0, :]
        ab = spec[1:, :]
        if avg_ab is None:
            avg_ab = np.mean(ab, axis=0)  # Get the mean of each column
        else:
            avg_ab = avg_ab
        ab_mc = ab - avg_ab  # 利用numpy数组的广播法则
        spec_mc = np.vstack((wavelength, ab_mc))

        return spec_mc

    def fit(self, spec):
        self.wavelength = spec[0, :]
        ab = spec[1:, :]
        self.avg_ab = np.mean(ab, axis=0)
        return self

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        ab = spec[1:, :]
        self.avg_ab = np.mean(ab, axis=0)
        spec_mc = self.mc(spec, avg_ab=self.avg_ab)
        return spec_mc

    def transform(self, input_data):
        '''
        用于当前实例
        :param input_data:
        :param avg_ab:
        :return:
        '''
        input_wavelength = input_data[0, :]
        spec_mc = self.mc(input_data, avg_ab=self.avg_ab)

        return spec_mc

    def inverse_transform(self, spec_mc, avg_ab=None):
        wavelength = spec_mc[0, :]
        ab_mc = spec_mc[1:, :]
        if avg_ab is None:
            ab_ori = ab_mc + self.avg_ab
        else:
            ab_ori = ab_mc + avg_ab
        spec_ori = np.vstack((wavelength, ab_ori))

        return spec_ori

class ZS(object):
    '''
    Zscore Standardization 中心标准化
    '''
    def __init__(self, avg_ab=None, std_ab=None):
        '''
        将原数据集各元素减去元素所在列的均值,再除以该列元素的标准差
        Centering using the average value, also called mean centering
        Scaling involves dividing the (centered) variables by individual measures of dispersion.
        Using the Standard Deviation as the scaling factor sets the variance for each variable to one,
        and is usually applied after mean centering.
        :param avg_ab:
        :param std_ab:
        '''
        self.avg_ab = avg_ab
        self.std_ab = std_ab
        return

    def zs(self, spec, avg_ab=None, std_ab=None):
        wavelength = spec[0, :]
        ab = spec[1:, :]
        if avg_ab is None and std_ab is None:
            ab_mean = np.mean(ab, axis=0)  # Get the mean of each column
            ab_mc = ab - ab_mean
            stdev = np.std(ab_mc, axis=0, ddof=1)
            ab_zs = ab_mc / stdev
        elif avg_ab is not None and std_ab is not None:
            ab_mean = avg_ab
            ab_mc = ab - ab_mean
            stdev = std_ab
            ab_zs = ab_mc / stdev
        spec_zs = np.vstack((wavelength, ab_zs))

        return spec_zs

    def fit(self, spec):
        self.wavelength = spec[0, :]
        ab = spec[1:, :]
        self.ab_mean = np.mean(ab, axis=0)
        self.ab_std = np.std(ab, axis=0, ddof=1)
        return self

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        ab = spec[1:, :]
        self.ab_mean = np.mean(ab, axis=0)
        self.ab_std = np.std(ab, axis=0, ddof=1)
        spec_zs = self.zs(spec, avg_ab=self.ab_mean, std_ab=self.ab_std)
        return spec_zs

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        spec_zs = self.zs(input_data, avg_ab=self.ab_mean, std_ab=self.ab_std)

        return spec_zs

    def inverse_transform(self, spec_as, avg_ab=None, std_ab=None):
        wavelength = spec_as[0, :]
        ab_zs = spec_as[1:, :]
        if avg_ab is None and std_ab is None:
            ab_ori = ab_zs * self.ab_std + self.ab_mean
        else:
            ab_ori = ab_zs * std_ab + avg_ab
        spec_ori = np.vstack((wavelength, ab_ori))

        return spec_ori

class MSC(object):
    '''
    Multiplicative Scatter Correction 多元散射校正
    '''
    def __init__(self, ideal_ab=None):
        self.ideal_ab = ideal_ab
        return

    def msc(self, spec, ideal_ab=None):
        wavelength = spec[0, :]
        ab = spec[1:, :]
        size_of_ab = ab.shape  # 10,700
        ab_msc = np.zeros(size_of_ab)  # 10,700
        # 对于校正集外的光谱进行MSC处理时则需要用到校正集样品的平均光谱ab_mean,
        # 即首先求取该光谱的c和d, 再进行MSC变换
        if ideal_ab is None:
            ab_mean = np.mean(ab, axis=0)  # 700,
        elif len(ideal_ab) != len(np.mean(ab, axis=0)):
            raise ValueError('数据点数不一致，输入参数有误！')
        else:
            ab_mean = ideal_ab
        for i in range(size_of_ab[0]):  # 求出每条光谱的c和d，c = b[0]   d = b[1]
            regression_coefficient = lsr(ab_mean, ab[i, :], order=1)['regression_coefficient']
            ab_msc[i, :] = (ab[i, :] - regression_coefficient[1]) / regression_coefficient[0]  # 利用广播法则

        spec_msc = np.vstack((wavelength, ab_msc))

        return spec_msc

    def fit(self, spec):
        self.wavelength = spec[0, :]
        ab = spec[1:, :]
        self.ideal_ab = np.mean(ab, axis=0)

        return self

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        ab = spec[1:, :]
        self.ideal_ab = np.mean(ab, axis=0)
        spec_msc = self.msc(spec, ideal_ab=self.ideal_ab)

        return spec_msc

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        spec_msc = self.msc(input_data, ideal_ab=self.ideal_ab)

        return spec_msc

class SGMSC(object):
    '''
     Savitzky-Golay + Multiplicative Scatter Correction 一阶导 + 多元散射校正
    '''
    def __init__(self, window_size=11, polyorder=2, deriv=1, ideal_ab=None):
        self.window_size = window_size
        self.polyorder = polyorder
        self.deriv = deriv
        self.ideal_ab = ideal_ab

        return

    def _msc(self, spec, ideal_ab=None):
        wavelength = spec[0, :]
        ab = spec[1:, :]
        size_of_ab = ab.shape  # 10,700
        ab_msc = np.zeros(size_of_ab)  # 10,700
        # 对于校正集外的光谱进行MSC处理时则需要用到校正集样品的平均光谱ab_mean,
        # 即首先求取该光谱的c和d, 再进行MSC变换
        if ideal_ab is None:
            ab_mean = np.mean(ab, axis=0)  # 700,
        elif len(ideal_ab) != len(np.mean(ab, axis=0)):
            raise ValueError('数据点数不一致，输入参数有误！')
        else:
            ab_mean = ideal_ab
        d_add = np.ones(size_of_ab[1])  # 700,   线性偏移量offset
        matrix_A = (np.vstack((ab_mean, d_add))).T  # (700,2)
        for i in range(size_of_ab[0]):  # 求出每条光谱的c和d，c = b[0]   d = b[1]
            b = dot(dot(np.linalg.inv(dot(matrix_A.T, matrix_A)), matrix_A.T), ab[i, :])
            ab_msc[i, :] = (ab[i, :] - b[1]) / b[0]  # 利用广播法则
        spec_msc = np.vstack((wavelength, ab_msc))

        return spec_msc

    def _sg(self, spec, window_size=11, polyorder=2, deriv=1):
        '''

        :param spec:
        :param window_size: must be odd and bigger than 2
        :param polyorder: must be bigger than deriv
        :param deriv:
        :return:
        '''
        try:
            window_size = np.abs(np.int(window_size))
            polyorder = np.abs(np.int(polyorder))
        except ValueError as msg:
            raise ValueError("window_size and polyorder have to be of type int")
        if window_size % 2 != 1 or window_size < 2:
            raise ValueError("window_size size must be a positive odd number")
        if window_size < polyorder:  # polyorder must be less than window_size
            raise ValueError("window_size is too small for the polynomials polyorder")
        if deriv > polyorder:  # 'deriv' must be less than or equal to 'polyorder'
            raise ValueError("请调小导数阶数！")

        n = spec.shape[0] - 1
        p = spec.shape[1]
        wavelength = spec[0, :]
        half_size = window_size // 2

        # 计算SG系数
        coef = np.zeros((window_size, polyorder + 1))
        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                coef[i, j] = np.power(i - int(window_size / 2), j)
        c = dot(inv(dot(coef.T, coef)), coef.T)

        # 拷贝SG系数
        coefs = np.zeros(window_size)  #(11,)
        for k in range(window_size):
            if deriv == 0 or deriv == 1:
                coefs[k] = c[deriv, k]
            elif deriv == 2:  # 需要调整系数
                coefs[k] = c[deriv, k] * 2
            elif deriv == 3:  # 需要调整系数
                coefs[k] = c[deriv, k] * 6
            elif deriv == 4:  # 需要调整系数
                coefs[k] = c[deriv, k] * 24

        # 处理吸光度
        tempdata = np.zeros((n, p))
        ab = spec[1:, :]
        for j in range(0, p - window_size + 1):
            data_window = ab[:, j:j + window_size]
            new_y = dot(data_window, coefs[:, np.newaxis])  # 将coefs增加到二维
            tempdata[:, j + half_size] = new_y.ravel()

        # 处理两端的数据
        for j in range(0, half_size):
            tempdata[:, j] = tempdata[:, half_size]
        for j in range(p - half_size, p):
            tempdata[:, j] = tempdata[:, p - half_size - 1]

        # 导数
        if deriv > 0:
            x_step = wavelength[1] - wavelength[0]
            x_step = np.power(x_step, deriv)
            ab_sg = tempdata / x_step
        else:
            ab_sg = tempdata

        spec_sg_matrix = np.vstack((wavelength, ab_sg))

        return spec_sg_matrix

    def sgmsc(self, spec, window_size=11, polyorder=2, deriv=1, ideal_ab=None):
        spec_sg = self._sg(spec, window_size=window_size, polyorder=polyorder, deriv=deriv)
        spec_sg_msc = self._msc(spec_sg, ideal_ab=ideal_ab)

        return spec_sg_msc

    def fit(self, spec):
        self.wavelength = spec[0, :]
        self.ideal_ab = np.mean(spec[1:, :], axis=0)

        return self

    def fit_transform(self, spec):
        spec_sg = self._sg(spec, window_size=self.window_size, polyorder=self.polyorder, deriv=self.deriv)
        self.fit(spec_sg)
        spec_sg_msc = self._msc(spec_sg, ideal_ab=self.ideal_ab)

        return spec_sg_msc

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        spec_sg = self._sg(input_data, window_size=self.window_size, polyorder=self.polyorder,
                                   deriv=self.deriv)
        spec_sg_msc = self._msc(spec_sg, ideal_ab=self.ideal_ab)

        return spec_sg_msc

# -------- 单样本操作 --------
class VN(object):
    '''
    Vector Normalization矢量归一化
    '''
    def __init__(self):
        return

    def vn(self, spec):
        wavelength = spec[0, :]
        ab = spec[1:, :]
        ab_vn = ab / np.linalg.norm(ab, axis=1, keepdims=True)
        spec_vn = np.vstack((wavelength, ab_vn))
        return spec_vn

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_vn = self.vn(spec)
        return spec_vn

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_vn = self.vn(input_data)

        return spec_vn

class SNV(object):
    '''
    Standard Normal Variate transformation 标准正态变换
    '''
    def __init__(self):
        return

    def snv(self, spec):
        wavelength = spec[0, :]
        ab = spec[1:, :]
        ab_mc = ab - np.mean(ab, axis=1, keepdims=True)
        ab_snv = ab_mc / np.std(ab, axis=1, keepdims=True, ddof=1)
        spec_snv = np.vstack((wavelength, ab_snv))
        return spec_snv

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_snv = self.snv(spec)
        return spec_snv

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_vn = self.snv(input_data)
        return spec_vn

class ECO(object):
    '''
    Eliminate Constant Offset 消除常数偏移量(减去各条光谱的最小值，使得最小值变成0)
    '''
    def __init__(self):
        return

    def eco(self, spec):
        wavelength = spec[0, :]
        ab = spec[1:, :]
        ab_sco = ab - np.min(ab, axis=1, keepdims=True)
        spec_sco = np.vstack((wavelength, ab_sco))
        return spec_sco

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_eco = self.eco(spec)
        return spec_eco

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_eco = self.eco(input_data)
        return spec_eco

class SSL(object):
    '''
    Subtract Straight Line 减去一条直线
    '''
    def __init__(self):
        return

    def ssl(self, spec):  # 必须含有波长
        wavelength = spec[0, :]
        ab = spec[1:, :]
        n_samples = ab.shape[0]
        ab_ssl = np.zeros(ab.shape)
        for i in range(n_samples):  # 求出趋势直线
            fit_value = lsr(wavelength, ab[i, :], order=1)['fit_value']
            ab_ssl[i, :] = ab[i, :] - fit_value.ravel()
        spec_ssl = np.vstack((wavelength, ab_ssl))

        return spec_ssl

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_ssl = self.ssl(spec)

        return spec_ssl

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_ssl = self.ssl(input_data)
        return spec_ssl

class DT(object):
    '''
    De-Trending 去趋势(2次多项式)
    '''
    def __init__(self):
        return

    def dt(self, spec):  # 必须含有波长
        wavelength = spec[0, :]
        ab = spec[1:, :]
        n_samples = ab.shape[0]
        ab_dt = np.zeros(ab.shape)
        for i in range(n_samples):  # 求出每条光谱的c和d，c = b[0]   d = b[1]
            fit_value = lsr(wavelength, ab[i, :], order=2)['fit_value']
            ab_dt[i, :] = ab[i, :] - fit_value.ravel()
        spec_dt = np.vstack((wavelength, ab_dt))

        return spec_dt

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_dt = self.dt(spec)

        return spec_dt

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_dt = self.dt(input_data)
        return spec_dt

class MMN(object):  # just used for spectra preprocessing
    '''
    Min-Max Normalization 最小最大归一化
    '''
    def __init__(self, norm_min=0, norm_max=1):
        self.norm_min = norm_min
        self.norm_max = norm_max
        return

    def mmn(self, spec, norm_min=0, norm_max=1):  # min max normalize
        wavelength = spec[0, :]
        ab = spec[1:, :]
        xmin = np.min(ab, axis=1, keepdims=True)
        xmax = np.max(ab, axis=1, keepdims=True)
        ab_mmn = norm_min + (ab - xmin) * (norm_max - norm_min) / (xmax - xmin)
        spec_mmn = np.vstack((wavelength, ab_mmn))
        return spec_mmn

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_mmn = self.mmn(spec, norm_min=self.norm_min, norm_max=self.norm_max)
        return spec_mmn

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_mmn = self.mmn(input_data, norm_min=self.norm_min, norm_max=self.norm_max)
        return spec_mmn

class SG(object):
    '''
    Savitzky-Golay 平滑与求导
    '''
    def __init__(self, window_size=11, polyorder=2, deriv=1):
        '''
        OPUS中polyorder默认为2
        :param window_size:
        :param polyorder:
        :param deriv:
        '''
        self.window_size = window_size
        self.polyorder = polyorder
        self.deriv = deriv
        return

    def sg(self, spec, window_size=11, polyorder=2, deriv=1):
        '''

        :param spec:
        :param window_size: must be odd and bigger than 2
        :param polyorder: must be bigger than deriv
        :param deriv:
        :return:
        '''
        try:
            window_size = np.abs(np.int(window_size))
            polyorder = np.abs(np.int(polyorder))
        except ValueError as msg:
            raise ValueError("window_size and polyorder have to be of type int")
        if window_size % 2 != 1 or window_size < 2:
            raise ValueError("window_size size must be a positive odd number")
        if window_size < polyorder:  # polyorder must be less than window_size
            raise ValueError("window_size is too small for the polynomials polyorder")
        if deriv > polyorder:  # 'deriv' must be less than or equal to 'polyorder'
            raise ValueError("请调小导数阶数！")

        n = spec.shape[0] - 1
        p = spec.shape[1]
        wavelength = spec[0, :]
        half_size = window_size // 2

        # 计算SG系数
        coef = np.zeros((window_size, polyorder + 1))
        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                coef[i, j] = np.power(i - int(window_size / 2), j)
        c = dot(inv(dot(coef.T, coef)), coef.T)

        # 拷贝SG系数
        coefs = np.zeros(window_size)  #(11,)
        for k in range(window_size):
            if deriv == 0 or deriv == 1:
                coefs[k] = c[deriv, k]
            elif deriv == 2:  # 需要调整系数
                coefs[k] = c[deriv, k] * 2
            elif deriv == 3:  # 需要调整系数
                coefs[k] = c[deriv, k] * 6
            elif deriv == 4:  # 需要调整系数
                coefs[k] = c[deriv, k] * 24

        # 处理吸光度
        tempdata = np.zeros((n, p))
        ab = spec[1:, :]
        for j in range(0, p - window_size + 1):
            data_window = ab[:, j:j + window_size]
            new_y = dot(data_window, coefs[:, np.newaxis])  # 将coefs增加到二维
            tempdata[:, j + half_size] = new_y.ravel()

        # 处理两端的数据
        for j in range(0, half_size):
            tempdata[:, j] = tempdata[:, half_size]
        for j in range(p - half_size, p):
            tempdata[:, j] = tempdata[:, p - half_size - 1]

        # 导数
        if deriv > 0:
            x_step = wavelength[1] - wavelength[0]
            x_step = np.power(x_step, deriv)
            ab_sg = tempdata / x_step
        else:
            ab_sg = tempdata

        spec_sg_matrix = np.vstack((wavelength, ab_sg))

        return spec_sg_matrix

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_sg = self.sg(spec, self.window_size, self.polyorder, self.deriv)
        return spec_sg

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_sg = self.sg(input_data, self.window_size, self.polyorder, self.deriv)
        return spec_sg

class SGSNV(object):
    '''
    Savitzky-Golay + SNV
    '''
    def __init__(self, window_size=11, polyorder=2, deriv=1):
        self.window_size = window_size
        self.polyorder = polyorder
        self.deriv = deriv
        return

    def _snv(self, spec):
        wavelength = spec[0, :]
        ab = spec[1:, :]
        ab_snv = (ab - np.mean(ab, axis=1, keepdims=True)) / np.std(ab, axis=1, keepdims=True, ddof=1)
        spec_snv = np.vstack((wavelength, ab_snv))
        return spec_snv

    def _sg(self, spec, window_size=11, polyorder=2, deriv=1):
        '''

        :param spec:
        :param window_size: must be odd and bigger than 2
        :param polyorder: must be bigger than deriv
        :param deriv:
        :return:
        '''
        try:
            window_size = np.abs(np.int(window_size))
            polyorder = np.abs(np.int(polyorder))
        except ValueError as msg:
            raise ValueError("window_size and polyorder have to be of type int")
        if window_size % 2 != 1 or window_size < 2:
            raise ValueError("window_size size must be a positive odd number")
        if window_size < polyorder:  # polyorder must be less than window_size
            raise ValueError("window_size is too small for the polynomials polyorder")
        if deriv > polyorder:  # 'deriv' must be less than or equal to 'polyorder'
            raise ValueError("请调小导数阶数！")

        n = spec.shape[0] - 1
        p = spec.shape[1]
        wavelength = spec[0, :]
        half_size = window_size // 2

        # 计算SG系数
        coef = np.zeros((window_size, polyorder + 1))
        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                coef[i, j] = np.power(i - int(window_size / 2), j)
        c = dot(inv(dot(coef.T, coef)), coef.T)

        # 拷贝SG系数
        coefs = np.zeros(window_size)  #(11,)
        for k in range(window_size):
            if deriv == 0 or deriv == 1:
                coefs[k] = c[deriv, k]
            elif deriv == 2:  # 需要调整系数
                coefs[k] = c[deriv, k] * 2
            elif deriv == 3:  # 需要调整系数
                coefs[k] = c[deriv, k] * 6
            elif deriv == 4:  # 需要调整系数
                coefs[k] = c[deriv, k] * 24

        # 处理吸光度
        tempdata = np.zeros((n, p))
        ab = spec[1:, :]
        for j in range(0, p - window_size + 1):
            data_window = ab[:, j:j + window_size]
            new_y = dot(data_window, coefs[:, np.newaxis])  # 将coefs增加到二维
            tempdata[:, j + half_size] = new_y.ravel()

        # 处理两端的数据
        for j in range(0, half_size):
            tempdata[:, j] = tempdata[:, half_size]
        for j in range(p - half_size, p):
            tempdata[:, j] = tempdata[:, p - half_size - 1]

        # 导数
        if deriv > 0:
            x_step = wavelength[1] - wavelength[0]
            x_step = np.power(x_step, deriv)
            ab_sg = tempdata / x_step
        else:
            ab_sg = tempdata

        spec_sg_matrix = np.vstack((wavelength, ab_sg))

        return spec_sg_matrix

    def sgsnv(self, spec, window_size=11, polyorder=2, deriv=1):
        spec_sg = self._sg(spec, window_size=window_size, polyorder=polyorder, deriv=deriv)
        spec_sg_snv = self._snv(spec_sg)

        return spec_sg_snv

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_sg = self._sg(spec, window_size=self.window_size, polyorder=self.polyorder, deriv=self.deriv)
        spec_sg_snv = self._snv(spec_sg)

        return spec_sg_snv

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_sg = self._sg(input_data, window_size=self.window_size, polyorder=self.polyorder, deriv=self.deriv)
        spec_sg_snv = self._snv(spec_sg)

        return spec_sg_snv

class SNVDT(object):
    '''
    SNV + DT
    '''
    def __init__(self):

        return

    def _snv(self, spec):
        wavelength = spec[0, :]
        ab = spec[1:, :]
        ab_snv = (ab - np.mean(ab, axis=1, keepdims=True)) / np.std(ab, axis=1, keepdims=True, ddof=1)
        spec_snv = np.vstack((wavelength, ab_snv))

        return spec_snv

    def _dt(self, spec):  # 必须含有波长
        wavelength = spec[0, :]
        ab = spec[1:, :]
        n_samples = ab.shape[0]
        ab_dt = np.zeros(ab.shape)
        for i in range(n_samples):  # 求出每条光谱的c和d，c = b[0]   d = b[1]
            fit_value = lsr(wavelength, ab[i, :], order=2)['fit_value']
            ab_dt[i, :] = ab[i, :] - fit_value.ravel()
        spec_dt = np.vstack((wavelength, ab_dt))

        return spec_dt

    def snvdt(self, spec):
        spec_snv = self._snv(spec)
        spec_snv_dt = self._dt(spec_snv)

        return spec_snv_dt

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_snv = self._snv(spec)
        spec_snv_dt = self._dt(spec_snv)

        return spec_snv_dt

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_snv = self._snv(input_data)
        spec_snv_dt = self._dt(spec_snv)

        return spec_snv_dt

class SGSSL(object):
    '''
    SG + SSL  求导 + 减去一条直线
    '''
    def __init__(self, window_size=11, polyorder=2, deriv=1):
        self.window_size = window_size
        self.polyorder = polyorder
        self.deriv = deriv
        return

    def _ssl(self, spec):  # 必须含有波长
        size_of_spec = spec.shape  # 第一行是x轴
        wavelength = spec[0, :]
        spec_ssl = np.zeros(size_of_spec)
        spec_ssl[0, :] = wavelength
        f_add = np.ones(size_of_spec[1])  # 用于构造A
        matrix_A = (np.vstack((wavelength, f_add))).T  # 2126 * 2
        for i in range(1, size_of_spec[0]):  # 从1开始，不算wavelength
            r = dot(dot(np.linalg.inv(dot(matrix_A.T, matrix_A)), matrix_A.T), spec[i, :])
            spec_ssl[i, :] = spec[i, :] - dot(matrix_A, r)
        return spec_ssl

    def _sg(self, spec, window_size=11, polyorder=2, deriv=1):
        '''

        :param spec:
        :param window_size: must be odd and bigger than 2
        :param polyorder: must be bigger than deriv
        :param deriv:
        :return:
        '''
        try:
            window_size = np.abs(np.int(window_size))
            polyorder = np.abs(np.int(polyorder))
        except ValueError as msg:
            raise ValueError("window_size and polyorder have to be of type int")
        if window_size % 2 != 1 or window_size < 2:
            raise ValueError("window_size size must be a positive odd number")
        if window_size < polyorder:  # polyorder must be less than window_size
            raise ValueError("window_size is too small for the polynomials polyorder")
        if deriv > polyorder:  # 'deriv' must be less than or equal to 'polyorder'
            raise ValueError("请调小导数阶数！")

        n = spec.shape[0] - 1
        p = spec.shape[1]
        wavelength = spec[0, :]
        half_size = window_size // 2

        # 计算SG系数
        coef = np.zeros((window_size, polyorder + 1))
        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                coef[i, j] = np.power(i - int(window_size / 2), j)
        c = dot(inv(dot(coef.T, coef)), coef.T)

        # 拷贝SG系数
        coefs = np.zeros(window_size)  #(11,)
        for k in range(window_size):
            if deriv == 0 or deriv == 1:
                coefs[k] = c[deriv, k]
            elif deriv == 2:  # 需要调整系数
                coefs[k] = c[deriv, k] * 2
            elif deriv == 3:  # 需要调整系数
                coefs[k] = c[deriv, k] * 6
            elif deriv == 4:  # 需要调整系数
                coefs[k] = c[deriv, k] * 24

        # 处理吸光度
        tempdata = np.zeros((n, p))
        ab = spec[1:, :]
        for j in range(0, p - window_size + 1):
            data_window = ab[:, j:j + window_size]
            new_y = dot(data_window, coefs[:, np.newaxis])  # 将coefs增加到二维
            tempdata[:, j + half_size] = new_y.ravel()

        # 处理两端的数据
        for j in range(0, half_size):
            tempdata[:, j] = tempdata[:, half_size]
        for j in range(p - half_size, p):
            tempdata[:, j] = tempdata[:, p - half_size - 1]

        # 导数
        if deriv > 0:
            x_step = wavelength[1] - wavelength[0]
            x_step = np.power(x_step, deriv)
            ab_sg = tempdata / x_step
        else:
            ab_sg = tempdata

        spec_sg_matrix = np.vstack((wavelength, ab_sg))

        return spec_sg_matrix

    def sgssl(self, spec, window_size=11, polyorder=2, deriv=1):
        spec_sg = self._sg(spec, window_size=window_size, polyorder=polyorder, deriv=deriv)
        spec_sg_ssl = self._ssl(spec_sg)

        return spec_sg_ssl

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_sg = self._sg(spec, window_size=self.window_size, polyorder=self.polyorder, deriv=self.deriv)
        spec_sg_ssl = self._ssl(spec_sg)

        return spec_sg_ssl

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_sg = self._sg(input_data, window_size=self.window_size, polyorder=self.polyorder, deriv=self.deriv)
        spec_sg_ssl = self._ssl(spec_sg)

        return spec_sg_ssl


# +++++++++++++++  Used for data +++++++++++++++

class MC4Data(object):
    def __init__(self, avg_data=None):
        self.avg_data = avg_data

        return

    def mc4data(self, data, avg_data=None):
        if avg_data is None:
            data_mean = np.mean(data, axis=0, keepdims=True)
        elif avg_data is not None:
            data_mean = avg_data
        data_mc = data - data_mean

        return data_mc

    def fit(self, data):
        self.avg_data = np.mean(data, axis=0, keepdims=True)
        return self

    def fit_transform(self, data):
        self.avg_data = np.mean(data, axis=0, keepdims=True)
        data_mc = self.mc4data(data, avg_data=self.avg_data)

        return data_mc

    def transform(self, input_data):
        data_mc = self.mc4data(input_data, avg_data=self.avg_data)
        return data_mc

    def inverse_transform(self, data_mc):
        data_ori = data_mc + self.avg_data
        return data_ori

class ZS4Data(object):
    def __init__(self, avg_data=None, std_data=None):
        self.avg_data = avg_data
        self.std_data = std_data
        return

    def zs4data(self, data, avg_data=None, std_data=None):
        if avg_data is None and std_data is None:
            data_mean = np.mean(data, axis=0)  # Get the mean of each column
            data_mc = data - data_mean
            data_stdev = np.std(data_mc, axis=0, ddof=1)
            data_zs = data_mc / data_stdev
        elif avg_data is not None and std_data is not None:
            data_mean = avg_data
            data_mc = data - data_mean
            data_stdev = std_data
            data_zs = data_mc / data_stdev

        return data_zs

    def fit(self, data):
        self.avg_data = np.mean(data, axis=0, keepdims=True)
        data_mc = data - self.avg_data
        self.std_data = np.std(data_mc, axis=0, ddof=1)

        return self

    def fit_transform(self, data):
        self.avg_data = np.mean(data, axis=0, keepdims=True)
        data_mc = data - self.avg_data
        self.std_data = np.std(data_mc, axis=0, ddof=1)
        data_zs = self.zs4data(data, avg_data=self.avg_data, std_data=self.std_data)

        return data_zs

    def transform(self, input_data):
        data_zs = self.zs4data(input_data, avg_data=self.avg_data, std_data=self.std_data)

        return data_zs

    def inverse_transform(self, data_zs):
        data_ori = data_zs * self.std_data + self.avg_data

        return data_ori

# +++++++++++++++ Function 用于最小二乘回归 +++++++++++++++
def generate_polynomial(X, order=1):
    if X.ndim == 1:
        X = X[:, np.newaxis]  # 如果一维数组，转成二维
    n_samples, n_variables = X.shape
    intercept = np.ones((n_samples, 1))  # offset 截距
    A = np.hstack((X, intercept))
    if order > 1:
        # 高次 ----> 低次
        for i in range(2, order+1):  # order==2
            s = X ** i
            A = np.hstack((s, A))

    return A

def lsr(X, y, order=1):  # 默认1次
    '''
    Least Square Regression 最小二乘回归
    :param X:
    :param y:
    :param order: 1,2,3... 适应多项式回归
    :return:
    regression_coefficient -
    fit_value -    fit_transform result (m X 1 column vector)
    residual -    residual   (m X 1 column vector)
    '''
    if X.ndim == 1:
        X = X[:, np.newaxis]  # 如果一维数组，转成二维
    if y.ndim == 1:
        y = y[:, np.newaxis]  # 如果一维数组，转成二维
    if X.shape[0] != y.shape[0]:
        raise ValueError('The number of samples is not equal!')
    n_samples = X.shape[0]
    intercept = np.ones((n_samples, 1))  # offset 截距
    A = generate_polynomial(X, order=order)
    regression_coefficient = dot(dot(inv(dot(A.T, A)), A.T), y)  # 系数(2,1)
    fit_value = dot(A, regression_coefficient)
    residual = fit_value - y

    return {'regression_coefficient':regression_coefficient,
            'fit_value':fit_value,
            'residual':residual}



# ================  Function 用于光谱列表操作，不在PLS中直接使用 ================

# +++++++++++++++ 多样本操作,针对列  +++++++++++++++

def msc_list(spec_list, ideal_ab=None):
    '''
    该函数用于MSC多元散射校正,目的与SNV基本相同,主要是消除颗粒分布不均匀及颗粒大小产生的散射影响。
    spec----光谱矩阵
    MSC针对一组光谱操作每条光谱都与平均光谱进行线性回归，spec_ori = c * spec_mean + d
    d----线性平移量(截距常数)
    c----倾斜偏移量(回归系数)
    CS*b = y  此处的y即为各原始光谱
    x=inv(CS'*CS)*CS'*b  求得的x包含c和d

    :param spec_list:
    :param ideal_ab: 0维数组，(1557,)
    :return:
    '''

    n = len(spec_list)
    if n == 1:
        raise ValueError('MSC针对多条光谱进行处理！')
    elif n > 1:
        result_list = []
        wavelength = spec_list[0][0, :]
        ab_list = [spec[1, :] for spec in spec_list]  # 此处ab (1557,)
        ab_array = np.array(ab_list)
        size_of_ab = ab_array.shape  # (10,1557)
        # 对于校正集外的光谱进行MSC处理时则需要用到校正集样品的平均光谱ab_mean,
        # 即首先求取该光谱的c和d, 再进行MSC变换
        if ideal_ab is None:
            ab_mean = np.mean(ab_array, axis=0)  # (1557,)
        elif len(ideal_ab) != len(np.mean(ab_array, axis=0)):
            raise ValueError('数据点数不一致，输入参数有误！')
        else:
            ab_mean = ideal_ab
        d_add = np.ones(size_of_ab[1])  # 700,   线性偏移量offset
        matrix_A = (np.vstack((ab_mean, d_add))).T  # (700,2)
        for i in range(n):  # 求出每条光谱的c和d，c = b[0]   d = b[1]
            b = dot(dot(np.linalg.inv(dot(matrix_A.T, matrix_A)), matrix_A.T), ab_array[i, :])
            ab_msc = (ab_array[i, :] - b[1]) / b[0]  # 利用广播法则
            spec_msc = np.vstack((wavelength, ab_msc))
            result_list.append(spec_msc)

    return result_list

def sgmsc_list(spec_list, window_size=11, polyorder=2, deriv=1, ideal_ab=None):
    '''
    先进行求导,再进行MSC
    :param spec:
    :param deriv:
    :param window_size:
    :param polyorder:
    :param ideal_ab:
    :return:
    '''
    n = len(spec_list)
    if n == 1:
        raise ValueError('MSC针对多条光谱进行处理！')
    elif n > 1:
        spec_sg_list = sg_list(spec_list, window_size=window_size, polyorder=polyorder, deriv=deriv)
        result_list = msc_list(spec_sg_list, ideal_ab=ideal_ab)

    return result_list

def mc_list(spec_list, avg_ab=None):
    '''
    该函数用于MC(mean-centering 均值中心化), 只返回均值中心化后的数据
    数据中心化通常是多变量数据建模的第一步，是指数据中的各个变量减去其均值所得的结果，
    以研究数据在其均值附近的变化，而不是数据的绝对值。
    根据实际问题的不同，有时亦使用其他的数值，而不一定是均值。

    :param spec:
    :param avg_ab: 0维数组
    :return:
    '''
    n = len(spec_list)
    if n == 1:
        raise ValueError('MC针对多条光谱进行处理！')
    elif n > 1:
        result_list = []
        wavelength = spec_list[0][0, :]
        ab_list = [spec[1, :] for spec in spec_list]  # 此处ab (1557,)
        ab_array = np.array(ab_list)
        if avg_ab is None:
            ab_array_mean = np.mean(ab_array, axis=0)  # Get the mean of each column
        else:
            ab_array_mean = avg_ab
        ab_array_mc = ab_array - ab_array_mean  # broadcast
        for i in range(n):
            spec_mc = np.vstack((wavelength, ab_array_mc[i, :]))
            result_list.append(spec_mc)

    return result_list

def zs_list(spec_list, avg_ab=None, std_ab=None):
    '''
    该函数用于AUTOSCALE(autoscaling, 标准化or均值方差化)，
    将原数据集各元素减去元素所在列的均值再除以该列元素的标准差。处理的结果：各列均值为0，方差为1
    又称：变量标度化，是对数据从变量方向的转换处理，包括二个方面，其一是中心化，其二是标度化。
    1. 数据中心化通常是多变量数据建模的第一步，是指数据中的各个变量减去其均值所得的结果，
    以研究数据在其均值附近的变化，而不是数据的绝对值。根据实际问题的不同，有时亦使用其他的数值，而不一定是均值。
    2. 数据标度化则是指数据除以其估计范围，比如标准偏差。当不同变量的相对数值范围相差很大时，标度化则尤为重要，
    其原因在于具有更大方差的变量，其在回归分析时影响亦越大
    :param spec_list:
    :param avg_ab: 0维数组
    :param std_ab: 0维数组
    :return:

    '''
    n = len(spec_list)
    if n == 1:
        raise ValueError('MC针对多条光谱进行处理！')
    elif n > 1:
        result_list = []
        wavelength = spec_list[0][0, :]
        ab_list = [spec[1, :] for spec in spec_list]  # 此处ab (1557,)
        ab_array = np.array(ab_list)
        if avg_ab is None and std_ab is None:
            ab_array_mean = np.mean(ab_array, axis=0)  # Get the mean of each column
            ab_array_mc = ab_array - ab_array_mean  # broadcast
            stdev = np.std(ab_array_mc, axis=0, ddof=1)
        elif avg_ab is not None and std_ab is not None:
            ab_array_mean = avg_ab
            ab_array_mc = ab_array - ab_array_mean
            stdev = std_ab
        ab_array_zs = ab_array_mc / stdev
        for i in range(n):
            spec_zs = np.vstack((wavelength, ab_array_zs[i, :]))
            result_list.append(spec_zs)

    return result_list

def avg_list(spec_list):  # average
    n = len(spec_list)
    if n == 1:
        raise ValueError('MC针对多条光谱进行处理！')
    elif n > 1:
        wavelength = spec_list[0][0, :]
        ab_list = [spec[1, :] for spec in spec_list]  # 此处ab (1557,)
        ab_array = np.array(ab_list)
        ab_array_mean = np.mean(ab_array, axis=0)
        result_spec = np.vstack((wavelength, ab_array_mean))

    return result_spec

# +++++++++++++++ 单样本操作,针对行  +++++++++++++++
def vn_list(spec_list):
    '''
    该函数用于VN(Vector Normalization)矢量归一化，目的是使数据具有相同长度，有效去除由于量测数值大小不同所导致的方差。
    VN一次针对一条光谱操作, 每条光谱的模长为1
    '''
    n = len(spec_list)
    result_list = []
    for i in range(n):
        spec = spec_list[i]
        wavelength = spec[0, :]
        ab = spec[1, :]  # (1557,)
        ab_vn = ab / np.linalg.norm(ab)
        spec_vn = np.vstack((wavelength, ab_vn))
        result_list.append(spec_vn)

    return result_list

def snv_list(spec_list):
    '''
    该函数用于SNV(Standardized Normal Variate transform)标准正态变换，
    主要是用来消除固体颗粒大小、表面散射以及光程变化对NIR漫反射光谱的影响
    SNV一次针对一条光谱操作(基于光谱阵的行)
    '''
    n = len(spec_list)
    result_list = []
    for i in range(n):
        spec = spec_list[i]
        wavelength = spec[0, :]
        ab = spec[1, :]
        ab_snv = (ab - np.mean(ab)) / np.std(ab, ddof=1)
        spec_snv = np.vstack((wavelength, ab_snv))
        result_list.append(spec_snv)

    return result_list

def eco_list(spec_list):
    '''
    该函数用于ECO(Eliminate Constant Offset)消除常数偏移量
    ECO对每条光谱分别操作
    '''
    n = len(spec_list)
    result_list = []
    for i in range(n):
        spec = spec_list[i]
        wavelength = spec[0, :]
        ab = spec[1, :]
        ab_sco = ab - np.min(ab)
        spec_sco = np.vstack((wavelength, ab_sco))
        result_list.append(spec_sco)

    return result_list

def ssl_list(spec_list):  # 必须含有波长
    '''
    ---与detrend中的linear同---
    该函数用于SSL(Subtract Straight Line)减去一条直线
    SSL一次针对一条光谱操作，每条光谱波长都与吸光度进行线性回归，吸光度ab = d * calset_wavelength_intersect + f
    CS*x = b   x=inv(CS'*CS)*CS'*b
    原始光谱减去这条拟合的直线
    :param spec: 含有波长数据的光谱矩阵，第0行存放x波长，1行存放吸光度
    :return:
    '''
    n = len(spec_list)
    result_list = []
    for i in range(n):
        spec = spec_list[i]
        wavelength = spec[0, :]
        ab = spec[1, :]  # (1557,)
        size_of_spec = spec.shape  # 第一行是x轴 (2, 1557)
        spec_ssl = np.zeros(size_of_spec)
        spec_ssl[0, :] = wavelength
        f_add = np.ones(size_of_spec[1])  # 用于构造A
        matrix_A = (np.vstack((wavelength, f_add))).T  # 1557 * 2
        r = dot(dot(np.linalg.inv(dot(matrix_A.T, matrix_A)), matrix_A.T), ab)
        spec_ssl[1:, :] = ab - dot(matrix_A, r)
        result_list.append(spec_ssl)

    return result_list

def mmn_list(spec_list, norm_min=0, norm_max=1):  # min max normalize
    n = len(spec_list)
    result_list = []
    for i in range(n):
        spec = spec_list[i]
        wavelength = spec[0, :]
        ab = spec[1, :]
        xmin = np.min(ab)
        xmax = np.max(ab)
        ab_mmn = norm_min + (ab - xmin) * (norm_max - norm_min) / (xmax - xmin)
        spec_mmn = np.vstack((wavelength, ab_mmn))
        result_list.append(spec_mmn)
    return result_list

def sg_list(spec_list, window_size=11, polyorder=2, deriv=1):
    '''

    :param spec_list:
    :param window_size: must be odd and bigger than 2
    :param polyorder: must be bigger than deriv
    :param deriv:
    :return:
    '''
    try:
        window_size = np.abs(np.int(window_size))
        polyorder = np.abs(np.int(polyorder))
    except ValueError as msg:
        raise ValueError("window_size and polyorder have to be of type int")
    if window_size % 2 != 1 or window_size < 2:
        raise ValueError("window_size size must be a positive odd number")
    if window_size < polyorder:  # polyorder must be less than window_size
        raise ValueError("window_size is too small for the polynomials polyorder")
    if deriv > polyorder:  # 'deriv' must be less than or equal to 'polyorder'
        raise ValueError("请调小导数阶数！")

    n = len(spec_list)
    half_size = window_size // 2
    result_list = []

    # 计算SG系数
    coef = np.zeros((window_size, polyorder+1))
    for i in range(coef.shape[0]):
        for j in range(coef.shape[1]):
            coef[i, j] = np.power(i - int(window_size / 2), j)
    c = dot(inv(dot(coef.T, coef)), coef.T)

    # 拷贝SG系数
    coefs = np.zeros(window_size)
    for k in range(window_size):
        if deriv == 2:  # 需要调整系数
            coefs[k] = c[deriv, k] * 2
        elif deriv == 3:  # 需要调整系数
            coefs[k] = c[deriv, k] * 6
        elif deriv == 4:  # 需要调整系数
            coefs[k] = c[deriv, k] * 24
        else:
            coefs[k] = c[deriv, k]
    # 处理吸光度
    for i in range(n):
        spec = spec_list[i]
        p = spec.shape[1]
        tempdata = np.zeros(p)
        wavelength = spec[0, :]
        ab = spec[1, :]
        p = spec.shape[1]
        for j in range(0, p-window_size+1):
            data_window = ab[j:j+window_size]
            new_y = inner(coefs, data_window)
            tempdata[j + half_size] = new_y

        # 处理两端的数据
        for j in range(0, half_size):
            tempdata[j] = tempdata[half_size]
        for j in range(p-half_size, p):
            tempdata[j] = tempdata[p - half_size - 1]

        # 导数
        if deriv > 0:
            x_step = wavelength[1] - wavelength[0]
            x_step = np.power(x_step, deriv)
            ab_sg = tempdata / x_step
        else:
            ab_sg = tempdata
        spec_sg = np.vstack((wavelength, ab_sg))
        result_list.append(spec_sg)

    return result_list

def sgsnv_list(spec_list, window_size=11, polyorder=2, deriv=1):
    '''

    :param spec_list:
    :param window_size:
    :param polyorder:
    :param deriv:
    :return:
    '''
    spec_sg_list = sg_list(spec_list, window_size, polyorder, deriv)
    result_list = snv_list(spec_sg_list)

    return result_list

def sgssl_list(spec_list, window_size=11, polyorder=2, deriv=1):
    '''

    :param spec_list:
    :param window_size:
    :param polyorder:
    :param deriv:
    :return:
    '''
    spec_sg_list = sg_list(spec_list, window_size, polyorder, deriv)
    result_list = ssl_list(spec_sg_list)

    return result_list


