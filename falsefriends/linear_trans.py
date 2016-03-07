# -*- coding: utf-8 -*

import numpy as np


# noinspection PyPep8Naming
def linear_transformation(X, Y):
    return np.linalg.lstsq(X, Y)[0]
