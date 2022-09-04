import scipy.sparse as sp
import numpy as np
import random
from time import time
def drop_scipy_matrix(matrix, keep_prob=0.5):
    print('doing drop_scipy_matrix')
    random_index = []
    s = time()
    shape = matrix.get_shape()
    nums = shape[0] * shape[1]
    prob_numpy = np.ones(nums)
    begin = 0
    end = 100000000
    while (nums - end > 0):
        temp_random_index = np.random.choice(range(begin, end), int((end - begin) * (1-keep_prob)), replace=False)
        random_index.extend(temp_random_index)
        begin = end
        end = end + 100000000
    temp_random_index = np.random.choice(range(begin, nums), int((nums - begin) * (1-keep_prob)), replace=False)
    random_index.extend(temp_random_index)
    for i in range(len(random_index)):
        prob_numpy[random_index[i]] = 0
    prob_numpy = prob_numpy.reshape(shape[0], shape[1])
    matrix = matrix.A
    matrix = np.multiply(matrix, prob_numpy)
    matrix = sp.csr_matrix(matrix)
    end = time()
    print(f"costing {end-s}s, done drop_scipy_matrix with keep_prob = {keep_prob}")
    return matrix


if __name__ == "__main__":
    row  = np.array([0, 0, 1, 3, 1, 0, 0, 1])
    col  = np.array([0, 2, 1, 3, 1, 0, 0, 2])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    a = sp.coo_matrix((data, (row, col)), shape=(4, 4))
    a = a.tolil()
    print(a)
    drop_scipy_matrix(a, keep_prob=0.5)
