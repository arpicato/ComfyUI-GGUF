import importlib
import sys
import types
import unittest
from pathlib import Path

import gguf
import torch


package = types.ModuleType("gguf_stream_test")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
GGMLTensor = importlib.import_module("gguf_stream_test.ops").GGMLTensor


class FileSourceTests(unittest.TestCase):
    def make_tensor(self):
        tensor = GGMLTensor(torch.tensor([1, 2], dtype=torch.uint8),
                            tensor_type=gguf.GGMLQuantizationType.Q8_0, tensor_shape=(2,))
        source = (object(), 0, 2)
        tensor.file_source = source
        tensor.file_source_version = tensor._version
        tensor.file_source_data_ptr = tensor.data_ptr()
        return tensor, source

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


if __name__ == "__main__":
    unittest.main()
