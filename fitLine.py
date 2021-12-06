
# https://blog.csdn.net/mago2015/article/details/84295425

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression


# 线性回归模型
df = pd.read_csv('dataset/line.csv', sep=',')
df.columns = ['X', 'y']
# print(df.head())
X = df[['X']].values
y = df[['y']].values
slr = LinearRegression()
slr.fit(X, y)
print("Slope: %.3f" % slr.coef_[0])
print("intercept: %.3f" % slr.intercept_)
plt.scatter(X, y, c='blue')
plt.plot(X, slr.predict(X), color='red')
plt.savefig('result/line.png')
plt.show()

# 使用RANSAC清除异常值高鲁棒对的线性回归模型
ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, residual_threshold=0.2, random_state=0)
ransac.fit(X, y)
in_mask = ransac.inlier_mask_
out_mask = np.logical_not(in_mask)
line_X = np.arange(1, 251, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[in_mask], y[in_mask], c='blue', marker='o', label='Inliers')
plt.scatter(X[out_mask], y[out_mask], c='green', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Index')
plt.ylabel('Height')
plt.savefig('result/ransac.png')
plt.show()

print(np.arange(1, 10, 1))  # [1 2 3 4 5 6 7 8 9]

X, y = make_regression(n_samples=200, n_features=1, noise=4.0, random_state=0)
reg = RANSACRegressor(random_state=0).fit(X, y)
reg.score(X, y)

print(X.shape)
print(y.shape)
print(X)
print(y)

plt.scatter(X, y, c='blue')
plt.plot(X, reg.predict(X), color='red')
plt.savefig('result/ran.png')
plt.show()
