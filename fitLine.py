
# https://blog.csdn.net/mago2015/article/details/84295425

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression


def linefit(x, y):
    # 线性回归--离散点拟合直线
    slr = LinearRegression()
    slr.fit(x, y)
    print("Coef: %.3f" % slr.coef_[0])  # 系数
    print("Intercept: %.3f" % slr.intercept_)  # 截距
    plt.scatter(x, y, c='blue')
    plt.plot(x, slr.predict(x), color='red')
    plt.savefig('result/line.png')
    plt.show()

def linefit_ransac(x, y):
    # ransac去除异常点后离散点拟合直线--迭代算法
    ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, residual_threshold=0.2, random_state=0)
    ransac.fit(x, y)
    inlier_mask = ransac.inlier_mask_  # 非异常点下标
    outlier_mask = np.logical_not(inlier_mask)  # 异常点下标
    plt.scatter(x[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
    plt.scatter(x[outlier_mask], y[outlier_mask], c='green', marker='s', label='Outliers')
    plt.plot(x, ransac.predict(x), color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('result/ransac.png')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('dataset/line.csv', sep=',')
    df.columns = ['X', 'y']
    print(df.head())
    X = df[['X']].values
    y = df[['y']].values
    # 最小二乘法线性回归
    linefit(X, y)
    # RANSAC最小二乘法线性回归
    linefit_ransac(X, y)
