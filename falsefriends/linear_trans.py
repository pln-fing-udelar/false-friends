# -*- coding: utf-8 -*

import numpy as np


# noinspection PyPep8Naming
def linear_transformation(origin_vectors, destination_vectors, backwards=False):
    if backwards:
        origin_vectors, destination_vectors = destination_vectors, origin_vectors
    return np.linalg.lstsq(origin_vectors, destination_vectors)[0]


def save_linear_transformation(file_name, transformation):
    np.savez(file_name, transformation)


def load_linear_transformation(file_name):
    return np.load(file_name)['arr_0']
