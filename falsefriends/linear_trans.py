# -*- coding: utf-8 -*

import numpy as np


# noinspection PyPep8Naming
def linear_transformation(X, Y):
    """Given a list of vectors X and a list of vectors Y returns the translation
       matrix
      :param X:origin language vectors
      :param Y:destination language vectors
    """
    return np.linalg.lstsq(X, Y)[0]
