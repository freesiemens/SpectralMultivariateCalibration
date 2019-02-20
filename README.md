SpectraAnalysis

Partial Least Squares implementation in Python
------------

### Provides
1. PartialLeastSquares(CrossValidation, ValsetValidation, Prediction)
* CrossValidation, cv
* ValsetValidation, vv
* Prediction, predict

###### It should be pointed out that before using 'predict', 'cv' or 'vv' must be run first.

2. Three PLS Algorithm:
* Improved Kernel Partial Least Squares, IKPLS
* Nonlinear Iterative Partial Least Squares，NIPALS
* Straightforward Implementation of a statistically inspired Modification of the Partial Least Squares, SIMPLS

3. Several Sampling Algorithm:
* montecarlo_sampling
* ks_sampling(Kennard-Stone)
* spxy_sampling

4. Several Samples split Algorithm:
* samples_systematic_split
* samples_ks_split
* samples_spxy_split
* samples_random_split

5. Popular Pretreat methods for Spectroscopy
* Multiplicative Scatter Correction 多元散射校正, MSC
* Multiplicative Scatter Correction + Savitzky-Golay 多元散射校正+求导, MSCSG
* Vector Normalization 矢量归一化, VN
* tandard Normal Variate transformation 标准正态变换, SNV
* Eliminate Constant Offset 消除常数偏移量, ECO
* Subtract Straight Line 减去一条直线, SSL
* De-Trending 去趋势, DT
* Min-Max Normalization 最小最大归一化, MMN
* Savitzky-Golay 平滑与求导, SG
* SNV + Savitzky-Golay, SNVSG
* SNV + DT, SNVDT
* SSL + SG, SSLSG
* Mean Centering 均值中心化, MC
* Zscore Standardization 标准化, ZS


Requirements
------------

Written using Python 3.6.3, numpy 1.15.4, scipy 1.1.0


Acknowledgement
------------

OPUS by Bruker, Unscrambler by CAMO, Pirouette by Infometrix, PLS_Toolbox by Eigenvector


References
----------
```
[1] Herschel W. XIII. Investigation of the powers of the prismatic colours to heat and illuminate objects; with remarks, that prove the different refrangibility of radiant heat. To which is added, an inquiry into the method of viewing the sun advantageously, with telescopes of large apertures and high magnifying powers[J]. Philosophical Transactions of the Royal Society of London, 1800, 90: 255-283.
[2] Wheeler O H. Near infrared spectra: A neglected field of spectral study[J]. Journal of Chemical Education, 1960, 37(5): 234.
[3] Haaland D M, Thomas E V. Partial least-squares methods for spectral analyses. 1. Relation to other quantitative calibration methods and the extraction of qualitative information[J]. Analytical Chemistry, 1988, 60(11): 1193-1202.
[4] Andersson M. A comparison of nine PLS1 algorithms[J]. Journal of Chemometrics, 2009, 23(10): 518-529.
[5] Wold S, Martens H, Wold H: The multivariate calibration problem in chemistry solved by the PLS method, Kågström B, Ruhe A, editor, Matrix Pencils: Proceedings of a Conference Held at Pite Havsbad, Sweden, March 22–24, 1982, Berlin, Heidelberg: Springer Berlin Heidelberg, 1983: 286-293.
[6] Strang G. Introduction to Linear Algebra[J], 2009.
[7] Geladi P, Kowalski B R. Partial least-squares regression: a tutorial[J]. Analytica Chimica Acta, 1986, 185: 1-17.
[8] De Jong S. SIMPLS: An alternative approach to partial least squares regression[J]. Chemometrics and Intelligent Laboratory Systems, 1993, 18(3): 251-263.
[9] Dayal B S, Macgregor J F. Improved PLS algorithms[J]. Journal of Chemometrics, 1997, 11(1): 73-85.
[10] Höskuldsson A. PLS regression methods[J]. Journal of Chemometrics, 1988, 2(3): 211-228.
[11] Lindgren F, Geladi P, Wold S. The kernel algorithm for PLS[J]. Journal of Chemometrics, 1993, 7(1): 45-59.
[12] De Jong S, Ter Braak C J F. Comments on the PLS kernel algorithm[J]. Journal of Chemometrics, 1994, 8(2): 169-174.
[13] Kennard R W, Stone L A. Computer Aided Design of Experiments[J]. Technometrics, 1969, 11(1): 137-148.
[14] Galvão R K H, Araujo M C U, José G E, et al. A method for calibration and validation subset partitioning[J]. Talanta, 2005, 67(4): 736-740.
[15] Moros J, Iñón F A, Garrigues S, et al. Determination of the energetic value of fruit and milk-based beverages through partial-least-squares attenuated total reflectance-Fourier transform infrared spectrometry[J]. Analytica Chimica Acta, 2005, 538(1): 181-193.
[16] Li B, Wang D, Xu C, et al. Flow-Injection Simultaneous Chemiluminescence Determination of Ascorbic Acid and L-Cysteine with Partial Least Squares Calibration[J]. Microchimica Acta, 2005, 149(3-4): 205-212.
[17] Savitzky A, Golay M J E. Smoothing and Differentiation of Data by Simplified Least Squares Procedures[J]. Analytical Chemistry, 1964, 36(8): 1627-1639.
[18] Barnes R J, Dhanoa M S, Lister S J. Standard Normal Variate Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra[J]. Applied Spectroscopy, 1989, 43(5): 772-777.
[19] Isaksson T, Næs T. The Effect of Multiplicative Scatter Correction (MSC) and Linearity Improvement in NIR Spectroscopy[J]. Applied Spectroscopy, 1988, 42(7): 1273-1284.
[20] Leardi R, Norgaard L. Sequential application of backward interval partial least squares and genetic algorithms for the selection of relevant spectral regions[J]. Journal of Chemometrics, 2004, 18(11): 486-497.
[21] Centner V, Massart D-L, De Noord O E, et al. Elimination of Uninformative Variables for Multivariate Calibration[J]. Analytical Chemistry, 1996, 68(21): 3851-3858.
[22] Cai W, Li Y, Shao X. A variable selection method based on uninformative variable elimination for multivariate calibration of near-infrared spectra[J]. Chemometrics and Intelligent Laboratory Systems, 2008, 90(2): 188-194.
[23] Han Q-J, Wu H-L, Cai C-B, et al. An ensemble of Monte Carlo uninformative variable elimination for wavelength selection[J]. Analytica Chimica Acta, 2008, 612(2): 121-125.
[24] Li H, Liang Y, Xu Q, et al. Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration[J]. Analytica Chimica Acta, 2009, 648(1): 77-84.
[25] Shenk J, Westerhaus M, Templeton W. Calibration transfer between near infrared reflectance spectrophotometers[J]. Crop science, 1985, 25(1): 159-161.
[26] Shenk J S, Westerhaus M O. Optical instrument calibration system. Google Patents, 1989.
[27] Wang Y, Veltkamp D J, Kowalski B R. Multivariate instrument standardization[J]. Analytical Chemistry, 1991, 63(23): 2750-2756.
[28] Wang Y, Kowalski B R. Calibration Transfer and Measurement Stability of Near-Infrared Spectrometers[J]. Applied Spectroscopy, 1992, 46(5): 764-771.
[29] Du W, Chen Z-P, Zhong L-J, et al. Maintaining the predictive abilities of multivariate calibration models by spectral space transformation[J]. Analytica Chimica Acta, 2011, 690(1): 64-70.
[30] Chen Z-P, Li L-M, Yu R-Q, et al. Systematic prediction error correction: A novel strategy for maintaining the predictive abilities of multivariate calibration models[J]. Analyst, 2011, 136(1): 98-106.
[31] Fan W, Liang Y, Yuan D, et al. Calibration model transfer for near-infrared spectra based on canonical correlation analysis[J]. Analytica Chimica Acta, 2008, 623(1): 22-29.
[32] Zheng K, Zhang X, Iqbal J, et al. Calibration transfer of near-infrared spectra for extraction of informative components from spectra with canonical correlation analysis[J]. Journal of Chemometrics, 2014, 28(10): 773-784.
[33] Chen W-R, Bin J, Lu H-M, et al. Calibration transfer via an extreme learning machine auto-encoder[J]. Analyst, 2016, 141(6): 1973-1980.
