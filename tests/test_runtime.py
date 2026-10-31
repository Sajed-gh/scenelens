import unittest
from unittest.mock import patch

from scenelens.runtime import resolve_device


class DeviceTests(unittest.TestCase):
    def test_cuda_is_default(self):
        with patch('torch.cuda.is_available',return_value=True), patch('torch.cuda.device_count',return_value=1):
            self.assertEqual(resolve_device(),'cuda:0')

    def test_unavailable_cuda_falls_back_to_cpu(self):
        with patch('torch.cuda.is_available',return_value=False):
            self.assertEqual(resolve_device(),'cpu')
            self.assertEqual(resolve_device('cuda:0'),'cpu')

    def test_explicit_cpu_override(self):
        with patch('torch.cuda.is_available',return_value=True):
            self.assertEqual(resolve_device('cpu'),'cpu')

    def test_invalid_gpu_index_is_reported(self):
        with patch('torch.cuda.is_available',return_value=True), patch('torch.cuda.device_count',return_value=1):
            with self.assertRaisesRegex(RuntimeError,'does not exist'):
                resolve_device('cuda:1')
