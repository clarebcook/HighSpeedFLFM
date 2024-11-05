import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def generate_A_matrix(order, X, Y, verbose=False):
    A = np.zeros(((order + 1) ** 2, len(X)))
    index = 0
    for j in (
        tqdm(range(order + 1), desc="Generating Matrix")
        if verbose
        else range(order + 1)
    ):
        for i in range(order + 1):
            row = np.power(X, j) * np.power(Y, i)
            A[index] = row
            index = index + 1
    return A.T


# https://stackoverflow.com/questions/33964913/equivalent-of-polyfit-for-a-2d-polynomial-in-python
def least_squares_fit(x, y, z, order=2, show=False):
    X = x.flatten()
    Y = y.flatten()
    Z = z.flatten()

    A = generate_A_matrix(order, X, Y)

    coeff, _, _, _ = np.linalg.lstsq(A, Z)
    z_predict = np.matmul(A, coeff)
    error = Z - z_predict
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        _ = ax.scatter(x, y, z)
        _ = ax.scatter(X, Y, z_predict)

        plt.xlabel("X")
        plt.ylabel("Y")

        plt.figure()
        plt.scatter(X, Y, c=error)
        plt.title("Fit Error")
        plt.colorbar()

    return coeff, A, error, z_predict

def generate_x_y_vectors(lengthx, lengthy):
    y, x = np.meshgrid(np.arange(lengthy), np.arange(lengthx))
    x = x.flatten() 
    y = y.flatten()
    return x, y

