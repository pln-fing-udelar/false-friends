#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from falsefriends import word_vectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_es = word_vectors.load_model('../resources/big/jairo/vectors_es.bin')

for word in model_es.index2word[:40000]:
    print(word)
