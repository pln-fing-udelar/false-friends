#!/usr/bin/env python3
# -*- coding: utf-8 -*

import argparse
import numpy as np


def pairwise(iterate):
    _iter = iter(iterate)
    return zip(_iter, _iter)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input', help="Input file", type=str)
    arg_parser.add_argument('output', help="Output file", type=str)
    args = arg_parser.parse_args()

    with open(args.input) as file:
        lines = file.readlines()

    X = []
    Y = []
    for line1, line2 in pairwise(lines):
        x = [float(coord) for coord in line1.split()]
        X.append(x)

        y = [float(coord) for coord in line2.split()]
        Y.append(y)

    transformation = np.linalg.lstsq(X, Y)[0]
    np.savetxt(args.output, transformation)
