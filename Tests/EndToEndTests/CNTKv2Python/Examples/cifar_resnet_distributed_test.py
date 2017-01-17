# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
import pytest
import subprocess
from cifar_convnet_distributed_test import mpiexec_test

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ResNet", "Python"))
script_under_test = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ResNet", "Python", "TrainResNet_CIFAR10_Distributed.py")

def test_cifar_convnet_distributed_mpiexec(device_id):
    params = [ "-e", "2"] # run only 2 epochs
    mpiexec_test(device_id, train_and_test_script, params, 0.5946, False, True)

def test_cifar_convnet_distributed_1bitsgd_mpiexec(device_id):
    params = ["-q", "1", "-e", "2"] # 2 epochs with 1BitSGD
    mpiexec_test(device_id, train_and_test_script, params, 0.5946, False, True)

def test_cifar_convnet_distributed_blockmomentum_mpiexec(device_id):
    params = ["-b", "32000", "-e", "2"] # 2 epochs with BlockMomentum SGD using blocksize 32000
    mpiexec_test(device_id, train_and_test_script, params, 0.55, True, False)
