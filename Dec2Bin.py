import numpy as np


def decimal2binary(decimal):
    # decimal: 十进制非负整数
    binary = ''
    if decimal == 0:
        binary += '0'
    while decimal > 0:
        remainder = np.remainder(decimal, 2)
        decimal = decimal // 2
        binary += str(remainder)
    return binary[::-1]


def binary2decimal(binary):
    # binary: 二进制数--string
    n = len(binary)
    decimal = 0
    for i in range(0, n):
        decimal += pow(2, n-i-1) * int(binary[i])
    return decimal


if __name__ == '__main__':
    print('##########')
    Dec = 134
    print(decimal2binary(Dec))
    Bin = str('1011')
    print(binary2decimal(Bin))
