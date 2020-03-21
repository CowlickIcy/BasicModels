import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

#%% 生成稀疏样本数据
np.random.seed(int(time.time()))

# 生成系统数据，样本50个，参数300维
n_samples, n_features = 50, 300

# 模拟正态分布
X = np.random.randn(n_samples, n_features)

# 每个变量对应的系数
coef = 2 * np.random.randn(n_features)

# 变量下标
inds = np.arange(n_features)

# 变量下标随机排列
np.random.shuffle(inds)

# 仅保留10个变量的参数，其他系数设置为零
coef[inds[10:]] = 0

# 得到目标值 ， y
y = np.dot(X, coef)

# 添加噪声
y += 0.01 * np.random.normal((n_samples, ))
#%% 将数据划分为训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=14)

# LassoCV:基于坐标下降法的Lasso交叉验证，这里使用20折
print('使用坐标轴下降法计算参数正则化路径')
model = LassoCV(cv=20).fit(X, y)

# 最终alpha的结果
m_log_alphas = -np.log10(model.alphas_)

# 作出交叉验证不同的alpha取值对应的MSE轨迹图
plt.figure()
ymin, ymax = 500, 1500
plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha:CV estimate')

plt.legend()
plt.xlabel('-log(alpha')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold:coordinate descent')
plt.axis('tight')
plt.ylim(ymin, ymax)

#%% LASSO回归的参数
alpha = model.alpha_
lasso = Lasso(max_iter=10000, alpha=alpha)

# 基于训练数据，得到的模型测试结果，这里使用坐标轴下降方法。
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
# 这里是R2可决系数（coefficient of determination）
# 回归平方和（RSS）在总变差（TSS）中所占的比重称为可决系数
# 可决系数可以作为综合度量回归模型对样本观测值拟合优度的度量指标。
# 可决系数越大，说明在总变差中由模型作出了解释的部分占的比重越大，模型拟合优度越好。
# 反之可决系数小，说明模型对样本观测值的拟合程度越差。
# R2可决系数最好的效果是1。
r2_score_lasso = r2_score(y_test, y_pred_lasso)

print("测试集上的r2可决系数：{:2f}".format(r2_score_lasso))

plt.plot(lasso.coef_, label='Lasso coefficients')
plt.plot(coef, '--', label='original coefficients')
plt.legend()

plt.show()

