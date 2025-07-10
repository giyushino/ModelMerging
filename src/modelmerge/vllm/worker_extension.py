# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

from trl.import_utils import (
    is_fastapi_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vllm_ascend_available,
    is_vllm_available,
)

if is_pydantic_available():
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn


if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator

from collections.abc import Sequence

class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` to handle
    efficient GPU-based communication using NCCL. The primary purpose of this class is to receive updated model weights
    from a client process and distribute them to all worker processes participating in model inference.
    """

    # The following attributes are initialized when `init_communicator` method is called.
    pynccl_comm = None  # Communicator for weight updates
    client_rank = None  # Source rank for broadcasting updated weights

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to
        communicate with vLLM workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
        """
        
        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        # Get the rank of the current worker in the global world group.
        try:
            rank = get_world_group().rank
        except Exception as e:
            raise

        # Create a stateless process group to manage communication between training processes and vLLM workers.
        try:
            pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        except Exception as e:
            raise

        # Initialize the NCCL-based communicator for weight synchronization.
        try:
            self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        except Exception as e:
            raise

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`str`):
                Data type of the weight tensor as a string (e.g., `"torch.float32"`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        # Convert dtype string to torch.dtype
        try:
            if isinstance(dtype, str):
                if dtype.startswith("torch."):
                    dtype_name = dtype.split(".")[-1]
                else:
                    dtype_name = dtype
                torch_dtype = getattr(torch, dtype_name)
            else:
                torch_dtype = dtype
        except Exception as e:
            raise

        # Allocate memory for incoming weights
        weight = torch.empty(shape, dtype=torch_dtype, device=self.device)

        # RECEIVE weights from client (client_rank)
        try:
            self.pynccl_comm.recv(weight, src=self.client_rank)
        except Exception as e:
            raise

        # Load weights into model
        try:
            self.model_runner.model.load_weights(weights=[(name, weight)])
        except Exception as e:
            raise

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """

        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None  # Ensure attribute is reset to None
            self.client_rank = None  # Ensure attribute is reset to None

    def test_method(self) -> str:
        """
        A simple test method to verify the worker extension is accessible.
        """
        return f"Worker extension test successful on device {self.device}"

    def simple_test(self) -> str:
        """
        A very simple test method that just returns a string.
        """
        return "Simple test successful!"
