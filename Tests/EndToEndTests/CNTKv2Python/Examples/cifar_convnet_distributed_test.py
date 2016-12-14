# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
import signal
import subprocess
import re
import pytest
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import set_default_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
train_and_test_script = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ConvNet", "Python", "ConvNet_CIFAR10_DataAug_Distributed.py")

TOLERANCE_ABSOLUTE = 2E-1
TIMEOUT_SECONDS = 300

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
def test_cifar_convnet_distributed_mpiexec(device_id):
<<<<<<< HEAD
=======
def mpiexec_test(device_id, train_and_test_script, params, expected_test_error):
>>>>>>> moved mpiexec call to a separate function in distributed ResNet and ConvNet tests
=======
def mpiexec_test(device_id, train_and_test_script, params, expected_test_error, allowTolerance):
>>>>>>> allow error difference tolerance for workers when using BlockMomentum SGD in tests
    if cntk_device(device_id).type() != DeviceKind_GPU:
       pytest.skip('test only runs on GPU')

    cmd = ["mpiexec", "-n", "2", "python", train_and_test_script] + params
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
<<<<<<< HEAD
    if sys.version_info[0] < 3:
        # TODO add timeout for Py2?
        out = p.communicate()[0]
    else:
        try:
            out = p.communicate(timeout=TIMEOUT_SECONDS)[0]  # in case we have a hang
        except subprocess.TimeoutExpired:
            os.kill(p.pid, signal.CTRL_C_EVENT)
            raise RuntimeError('Timeout in mpiexec, possibly hang')
    str_out = out.decode(sys.getdefaultencoding())
    pdb.set_trace()
<<<<<<< HEAD
=======
    results = re.findall("Final Results: Minibatch\[.+?\]: errs = (.+?)%", str_out)
    assert len(results) == 2
    assert results[0] == results[1]
    expected_test_error = 0.617
    assert np.allclose(float(results[0])/100, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)
=======
>>>>>>> moved mpiexec call to a separate function in distributed ResNet and ConvNet tests

    try:
        out = p.communicate(timeout=TIMEOUT_SECONDS)[0]  # in case we have a hang
    except subprocess.TimeoutExpired:
        os.kill(p.pid, signal.CTRL_C_EVENT)
        raise RuntimeError('Timeout in mpiexec, possibly hang')

    str_out = out.decode(sys.getdefaultencoding())
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> tests ResNet and ConvNet restructure
=======
    pdb.set_trace()
=======
>>>>>>> allow error difference tolerance for workers when using BlockMomentum SGD in tests
    results = re.findall("Final Results: Minibatch\[.+?\]: errs = (.+?)%", str_out)

    assert len(results) == 2
 
<<<<<<< HEAD
    if "-b" not in params:
        assert results[0] == results[1]
    else:
        assert np.allclose(float(results[0])/100, float(results[1])/100,
                       atol=TOLERANCE_ABSOLUTE)

    assert np.allclose(float(results[0])/100, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)

def test_cifar_convnet_distributed_mpiexec(device_id):
   
    params = [ "-e", "2"] # run only 2 epochs
    mpiexec_test(device_id, train_and_test_script, params, 0.617)

def test_cifar_convnet_distributed_1bitsgd_mpiexec(device_id):
    
    params = ["-q", "1", "-e", "2"] # 2 epochs with 1BitSGD
    mpiexec_test(device_id, train_and_test_script, params, 0.617)

<<<<<<< HEAD
    cmd = ["mpiexec", "-n", "2", "python", os.path.join(abs_path, "run_cifar_convnet_distributed.py")]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        out = p.communicate(timeout=TIMEOUT_SECONDS)[0]  # in case we have a hang
    except subprocess.TimeoutExpired:
        os.kill(p.pid, signal.CTRL_C_EVENT)
        raise RuntimeError('Timeout in mpiexec, possibly hang')
    str_out = out.decode(sys.getdefaultencoding())
>>>>>>> tests ResNet and ConvNet restructure
    results = re.findall("Final Results: Minibatch\[.+?\]: errs = (.+?)%", str_out)
    assert len(results) == 2
    assert results[0] == results[1]
    expected_test_error = 0.617
    assert np.allclose(float(results[0])/100, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)

def test_cifar_convnet_distributed_1bitsgd_mpiexec(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')

    cmd = ["mpiexec", "-n", "2", "python", os.path.join(abs_path, "run_cifar_convnet_distributed.py")]
=======
   
    skip_if_cpu(device_id)
    cmd = ["mpiexec", "-n", "2", "python", train_and_test_script, "-e", "2"]
>>>>>>> enabled for pytest ConvNet and ResNet scripts from Examples/Image/Classification
=======
def mpiexec_test(device_id, train_and_test_script, params, expected_test_error):
    if cntk_device(device_id).type() != DeviceKind_GPU:
       pytest.skip('test only runs on GPU')

    cmd = ["mpiexec", "-n", "2", "python", train_and_test_script] + params
>>>>>>> moved mpiexec call to a separate function in distributed ResNet and ConvNet tests
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    try:
        out = p.communicate(timeout=TIMEOUT_SECONDS)[0]  # in case we have a hang
    except subprocess.TimeoutExpired:
        os.kill(p.pid, signal.CTRL_C_EVENT)
        raise RuntimeError('Timeout in mpiexec, possibly hang')

    str_out = out.decode(sys.getdefaultencoding())
    results = re.findall("Final Results: Minibatch\[.+?\]: errs = (.+?)%", str_out)

    assert len(results) == 2
 
    if "-b" not in params:
=======
    if not allowTolerance:
>>>>>>> allow error difference tolerance for workers when using BlockMomentum SGD in tests
        assert results[0] == results[1]
    else:
        assert np.allclose(float(results[0])/100, float(results[1])/100,
                       atol=TOLERANCE_ABSOLUTE)

    assert np.allclose(float(results[0])/100, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)

def test_cifar_convnet_distributed_mpiexec(device_id):
   
    params = [ "-e", "2"] # run only 2 epochs
    mpiexec_test(device_id, train_and_test_script, params, 0.617, False)

def test_cifar_convnet_distributed_1bitsgd_mpiexec(device_id):
    
    params = ["-q", "1", "-e", "2"] # 2 epochs with 1BitSGD
    mpiexec_test(device_id, train_and_test_script, params, 0.617, False)

<<<<<<< HEAD
    cmd = ["mpiexec", "-n", "2", "python", train_and_test_script, "-q", "1", "-e", "2"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        out = p.communicate(timeout=TIMEOUT_SECONDS)[0]  # in case we have a hang
    except subprocess.TimeoutExpired:
        os.kill(p.pid, signal.CTRL_C_EVENT)
        raise RuntimeError('Timeout in mpiexec, possibly hang')
    str_out = out.decode(sys.getdefaultencoding())
    results = re.findall("Final Results: Minibatch\[.+?\]: errs = (.+?)%", str_out)
    assert len(results) == 2
    assert results[0] == results[1]
    expected_test_error = 0.617
<<<<<<< HEAD
=======
    assert np.allclose(float(results[0])/100, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)

def test_cifar_convnet_distributed_blockmomentum_mpiexec(device_id):
    skip_if_cpu(device_id)

    cmd = ["mpiexec", "-n", "2", "python", train_and_test_script, "-b", "32000", "-e", "2"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        out = p.communicate(timeout=TIMEOUT_SECONDS)[0]  # in case we have a hang
    except subprocess.TimeoutExpired:
        os.kill(p.pid, signal.CTRL_C_EVENT)
        raise RuntimeError('Timeout in mpiexec, possibly hang')
    str_out = out.decode(sys.getdefaultencoding())
    results = re.findall("Final Results: Minibatch\[.+?\]: errs = (.+?)%", str_out)
    assert len(results) == 2
    assert np.allclose(float(results[0])/100, float(results[1])/100,
                       atol=TOLERANCE_ABSOLUTE)
    expected_test_error = 0.6457
>>>>>>> enabled for pytest ConvNet and ResNet scripts from Examples/Image/Classification
    assert np.allclose(float(results[0])/100, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)
=======

def test_cifar_convnet_distributed_blockmomentum_mpiexec(device_id):
>>>>>>> moved mpiexec call to a separate function in distributed ResNet and ConvNet tests

    params = ["-b", "32000", "-e", "2"] # 2 epochs with BlockMomentum SGD using blocksize 32000
<<<<<<< HEAD
    mpiexec_test(device_id, train_and_test_script, params, 0.6457)
    
=======
def test_cifar_convnet_distributed_blockmomentum_mpiexec(device_id):

    params = ["-b", "32000", "-e", "2"] # 2 epochs with BlockMomentum SGD using blocksize 32000
    mpiexec_test(device_id, train_and_test_script, params, 0.6457)
    
>>>>>>> moved mpiexec call to a separate function in distributed ResNet and ConvNet tests
=======
    mpiexec_test(device_id, train_and_test_script, params, 0.6457, True)
    
>>>>>>> allow error difference tolerance for workers when using BlockMomentum SGD in tests
