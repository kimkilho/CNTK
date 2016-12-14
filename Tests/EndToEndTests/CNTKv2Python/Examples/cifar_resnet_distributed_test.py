# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import set_default_device
from cntk.io import FULL_DATA_SWEEP
from cntk import distributed
import pytest
import subprocess

abs_path = os.path.dirname(os.path.abspath(__file__))
<<<<<<< HEAD
<<<<<<< HEAD
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ResNet", "Python"))
from TrainResNet_CIFAR10_Distributed import train_and_evaluate, create_reader
=======
sys.path.append(abs_path)
from cifar_convnet_distributed_test import mpiexec_test

train_and_test_script = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ResNet", "Python", "TrainResNet_CIFAR10_Distributed.py")
>>>>>>> enabled for pytest ConvNet and ResNet scripts from Examples/Image/Classification

TOLERANCE_ABSOLUTE = 2E-1

<<<<<<< HEAD
<<<<<<< HEAD
def test_cifar_resnet_distributed_error(device_id, is_1bit_sgd):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    set_default_device(cntk_device(device_id))

    if not is_1bit_sgd:
        pytest.skip('test only runs in 1-bit SGD')

=======
def test_cifar_resnet_distributed_mpiexec(device_id):
    skip_if_cpu(device_id)

    cmd = ["mpiexec", "-n", "2", "python", train_and_test_script, "-e", "2"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
>>>>>>> enabled for pytest ConvNet and ResNet scripts from Examples/Image/Classification
    try:
        base_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"Image/CIFAR/v0/cifar-10-batches-py".split("/"))
    except KeyError:
        base_path = os.path.join(
            *"../../../../Examples/Image/DataSets/CIFAR-10".split("/"))

    base_path = os.path.normpath(base_path)
    os.chdir(os.path.join(base_path, '..'))

<<<<<<< HEAD
    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    set_computation_network_trace_level(1)
    set_fixed_random_seed(1)  # BUGBUG: has no effect at present  # TODO: remove debugging facilities once this all works
    #force_deterministic_algorithms()
    # TODO: do the above; they lead to slightly different results, so not doing it for now

    distributed_learner_factory = lambda learner: distributed.data_parallel_distributed_learner(
        learner=learner,
        num_quantization_bits=32,
        distributed_after=0)
=======
def test_cifar_resnet_distributed_1bitsgd_mpiexec(device_id):
    skip_if_cpu(device_id)

    cmd = ["mpiexec", "-n", "2", "python", os.path.join(abs_path, "run_cifar_resnet_distributed.py"),"-e", "2", "-q", "1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        out = p.communicate(timeout=TIMEOUT_SECONDS)[0]  # in case we have a hang
>>>>>>> enabled for pytest ConvNet and ResNet scripts from Examples/Image/Classification
=======
sys.path.append(abs_path)
from cifar_convnet_distributed_test import mpiexec_test

train_and_test_script = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ResNet", "Python", "TrainResNet_CIFAR10_Distributed.py")

TOLERANCE_ABSOLUTE = 2E-1
TIMEOUT_SECONDS = 300

def test_cifar_convnet_distributed_mpiexec(device_id):
   
    params = [ "-e", "2"] # run only 2 epochs
    mpiexec_test(device_id, train_and_test_script, params, 0.5946, False)

def test_cifar_convnet_distributed_1bitsgd_mpiexec(device_id):
    
<<<<<<< HEAD
    params = ["-q", "1", "-e", "2"] # 2 epochs with 1BitSGD
    mpiexec_test(device_id, train_and_test_script, params, 0.5946)
>>>>>>> moved mpiexec call to a separate function in distributed ResNet and ConvNet tests


<<<<<<< HEAD
<<<<<<< HEAD
    test_error = train_and_evaluate(reader_train_factory, test_reader, 'resnet20', 5, distributed_learner_factory)

    expected_test_error = 0.282
=======
def test_cifar_resnet_distributed_blockmomentum_mpiexec(device_id):
    skip_if_cpu(device_id)

    cmd = ["mpiexec", "-n", "2", "python", os.path.join(abs_path, "run_cifar_resnet_distributed.py"), "-e", "2", "-b", "32000"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        out = p.communicate(timeout=TIMEOUT_SECONDS)[0]  # in case we have a hang
>>>>>>> enabled for pytest ConvNet and ResNet scripts from Examples/Image/Classification
=======
def test_cifar_convnet_distributed_blockmomentum_mpiexec(device_id):

    params = ["-b", "32000", "-e", "2"] # 2 epochs with BlockMomentum SGD using blocksize 32000
    mpiexec_test(device_id, train_and_test_script, params, 0.55)
>>>>>>> moved mpiexec call to a separate function in distributed ResNet and ConvNet tests
>>>>>>> moved mpiexec call to a separate function in distributed ResNet and ConvNet tests

    assert np.allclose(test_error, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)
<<<<<<< HEAD
    distributed.Communicator.finalize()
=======
=======
def test_cifar_convnet_distributed_mpiexec(device_id):
   
    params = [ "-e", "2"] # run only 2 epochs
    mpiexec_test(device_id, train_and_test_script, params, 0.5946)

def test_cifar_convnet_distributed_1bitsgd_mpiexec(device_id):
    
=======
>>>>>>> allow error difference tolerance for workers when using BlockMomentum SGD in tests
    params = ["-q", "1", "-e", "2"] # 2 epochs with 1BitSGD
    mpiexec_test(device_id, train_and_test_script, params, 0.5946, False)

>>>>>>> moved mpiexec call to a separate function in distributed ResNet and ConvNet tests

def test_cifar_convnet_distributed_blockmomentum_mpiexec(device_id):

    params = ["-b", "32000", "-e", "2"] # 2 epochs with BlockMomentum SGD using blocksize 32000
    mpiexec_test(device_id, train_and_test_script, params, 0.55, True)

>>>>>>> enabled for pytest ConvNet and ResNet scripts from Examples/Image/Classification
