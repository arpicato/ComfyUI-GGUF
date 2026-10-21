import importlib
import os
import sys
import tempfile
import types
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import gguf
import numpy as np
import torch


package = types.ModuleType("gguf_stream_test")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
ops = importlib.import_module("gguf_stream_test.ops")
GGMLTensor = ops.GGMLTensor


class FileSourceTests(unittest.TestCase):
    def make_tensor(self):
        tensor = GGMLTensor(torch.tensor([1, 2], dtype=torch.uint8),
                            tensor_type=gguf.GGMLQuantizationType.Q8_0, tensor_shape=(2,))
        tensor.set_file_source(object(), 0, 2, readonly=False)
        return tensor, tensor.file_source

    def test_original_tensor_can_stream(self):
        tensor, source = self.make_tensor()
        self.assertIs(tensor.current_file_source(), source)

    def test_modified_tensor_uses_its_memory(self):
        tensor, _ = self.make_tensor()
        tensor.add_(1)
        self.assertIsNone(tensor.current_file_source())

    def test_replaced_storage_uses_its_memory(self):
        tensor, _ = self.make_tensor()
        tensor.data = torch.tensor([3, 4], dtype=torch.uint8)
        self.assertIsNone(tensor.current_file_source())

    def test_cpu_and_dtype_conversion_use_regular_transfer(self):
        tensor, _ = self.make_tensor()
        before = GGMLTensor.streamed_bytes
        self.assertEqual(tensor.to("cpu").tolist(), [1, 2])
        self.assertEqual(tensor.to(dtype=torch.float32).tolist(), [1.0, 2.0])
        self.assertEqual(GGMLTensor.streamed_bytes, before)

    def test_readonly_inference_tensor_can_stream(self):
        data = np.array([1, 2], dtype=np.uint8)
        data.flags.writeable = False
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            tensor = GGMLTensor(torch.from_numpy(data),
                                tensor_type=gguf.GGMLQuantizationType.Q8_0, tensor_shape=(2,))
            tensor.set_file_source(object(), 0, 2, readonly=True)
            self.assertIs(tensor.current_file_source(), tensor.file_source)

    def test_writable_inference_tensor_cannot_stream(self):
        with torch.inference_mode():
            tensor = GGMLTensor(torch.tensor([1, 2], dtype=torch.uint8),
                                tensor_type=gguf.GGMLQuantizationType.Q8_0, tensor_shape=(2,))
            tensor.set_file_source(object(), 0, 2, readonly=False)
            self.assertIsNone(tensor.current_file_source())

    def test_staging_allocation_failure_is_cached(self):
        device = torch.device("cuda:7")
        with mock.patch.object(ops, "_GGUFStagingBuffers", side_effect=RuntimeError("pinning unavailable")):
            self.assertIsNone(ops._gguf_stager(device))
            self.assertIsNone(ops._gguf_stager(device))
        self.assertIsNone(ops._gguf_stagers.pop(device))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA or ROCm device")
class DeviceTransferTests(unittest.TestCase):
    def test_multi_chunk_transfer_and_reload(self):
        size = 3 * 8 * 1024 * 1024 + 13
        payload = np.arange(size, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"header" + payload.tobytes())
            mapped = np.memmap(path, mode="r", dtype=np.uint8, offset=6, shape=(size,))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                tensor = GGMLTensor(torch.from_numpy(mapped),
                                    tensor_type=gguf.GGMLQuantizationType.Q8_0, tensor_shape=(size,))
            fd = os.open(path, os.O_RDONLY)
            try:
                tensor.set_file_source(types.SimpleNamespace(fd=fd), 6, size, readonly=True)
                before = GGMLTensor.streamed_bytes
                gpu = tensor.to("cuda")
                self.assertEqual(GGMLTensor.streamed_bytes - before, size)
                self.assertTrue(np.array_equal(gpu.to("cpu").numpy(), payload))
                self.assertTrue(np.array_equal(tensor.to("cuda").to("cpu").numpy(), payload))
                tensor.patches = [("synthetic",)]
                self.assertEqual(tensor.to("cuda").patches, tensor.patches)
            finally:
                os.close(fd)

    def test_concurrent_transfers_share_staging_safely(self):
        payload = np.arange(9 * 1024 * 1024 + 3, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(payload.tobytes())
            mapped = np.memmap(path, mode="r", dtype=np.uint8, shape=payload.shape)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                tensor = GGMLTensor(torch.from_numpy(mapped),
                                    tensor_type=gguf.GGMLQuantizationType.Q8_0,
                                    tensor_shape=payload.shape)
            fd = os.open(path, os.O_RDONLY)
            try:
                tensor.set_file_source(types.SimpleNamespace(fd=fd), 0, payload.size, readonly=True)
                with ThreadPoolExecutor(max_workers=3) as pool:
                    copies = list(pool.map(lambda _: tensor.to("cuda").to("cpu").numpy(), range(3)))
                self.assertTrue(all(np.array_equal(copy, payload) for copy in copies))
            finally:
                os.close(fd)

    def test_modified_tensor_falls_back_and_survives_offload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(bytes([1, 2, 3, 4]))
            fd = os.open(path, os.O_RDONLY)
            try:
                tensor = GGMLTensor(torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
                                    tensor_type=gguf.GGMLQuantizationType.Q8_0, tensor_shape=(4,))
                tensor.set_file_source(types.SimpleNamespace(fd=fd), 0, 4, readonly=False)
                tensor.add_(1)
                before = GGMLTensor.streamed_bytes
                gpu = tensor.to("cuda")
                self.assertEqual(GGMLTensor.streamed_bytes, before)
                self.assertEqual(gpu.to("cpu").tolist(), [2, 3, 4, 5])
                self.assertEqual(gpu.to("cpu").to("cuda").to("cpu").tolist(), [2, 3, 4, 5])
            finally:
                os.close(fd)

    def test_staging_unavailable_uses_regular_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(bytes([1, 2, 3, 4]))
            fd = os.open(path, os.O_RDONLY)
            try:
                tensor = GGMLTensor(torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
                                    tensor_type=gguf.GGMLQuantizationType.Q8_0, tensor_shape=(4,))
                tensor.set_file_source(types.SimpleNamespace(fd=fd), 0, 4, readonly=False)
                before = GGMLTensor.streamed_bytes
                with mock.patch.object(ops, "_gguf_stager", return_value=None):
                    gpu = tensor.to("cuda")
                self.assertEqual(GGMLTensor.streamed_bytes, before)
                self.assertEqual(gpu.to("cpu").tolist(), [1, 2, 3, 4])
            finally:
                os.close(fd)


if __name__ == "__main__":
    unittest.main()
