# SpectraAnalysis
#### Partial Least Squares implementation in Python
> Provides
1. PartialLeastSquares(CrossValidation, ValsetValidation, Prediction)
## -- CrossValidation, cv
## -- ValsetValidation, vv
##-- Prediction, predict
#### ++ It should be pointed out that before using 'predict', 'cv' or 'vv' must be run first.

2. Three PLS Algorithm:
## -- Improved Kernel Partial Least Squares, IKPLS
## -- Nonlinear Iterative Partial Least Squares，NIPALS
## -- Straightforward Implementation of a statistically inspired Modification of the Partial Least Squares, SIMPLS

3. Several Sampling Algorithm:
## -- montecarlo_sampling
## -- ks_sampling(Kennard-Stone)
## -- spxy_sampling

4. Several Samples split Algorithm:
## -- samples_systematic_split
## -- samples_ks_split
## -- samples_spxy_split
## -- samples_random_split

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
