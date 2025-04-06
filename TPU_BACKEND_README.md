# TPU Backend for llama-cpp-python

This is a TPU backend for llama-cpp-python that allows you to run LLM inference on Google Cloud TPUs.

## Prerequisites

- Python 3.8 or higher
- llama-cpp-python
- PyTorch
- PyTorch/XLA
- CMake
- C++ compiler (GCC, Clang, or MSVC)

## Installation

### 1. Install PyTorch and PyTorch/XLA

First, you need to install PyTorch and PyTorch/XLA. Follow the instructions on the [PyTorch/XLA GitHub page](https://github.com/pytorch/xla).

For TPU VMs, you can use:

```bash
pip install torch~=2.6.0 'torch_xla[tpu]~=2.6.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### 2. Install llama-cpp-python

Install llama-cpp-python with:

```bash
pip install llama-cpp-python
```

### 3. Build and Install the TPU Backend

Run the setup script:

```bash
python setup_tpu_backend.py
```

This will build and install the TPU backend for llama-cpp-python.

## Usage

### Command-line Interface

You can use the TPU backend with the HelpingAI Inference Server:

```bash
python test.py --model your-model-path --use-tpu --tpu-layers -1 --tpu-bf16
```

Command-line options:

- `--use-tpu`: Enable TPU support
- `--tpu-cores`: Number of TPU cores to use (default: 8)
- `--tpu-layers`: Number of layers to offload to TPU (-1 means all)
- `--tpu-bf16`: Use bfloat16 precision for TPU (recommended)
- `--tpu-memory-limit`: Memory limit for TPU tensor allocator (e.g., '1GB', '2GB')

### Python API

```python
from llama_cpp import Llama

# Initialize the model with TPU support
model = Llama(
    model_path="path/to/model.gguf",
    use_tpu=True,
    n_gpu_layers=-1  # Use all layers on TPU
)

# Generate text
output = model("What is the capital of France?", max_tokens=50)
print(output["choices"][0]["text"])
```

## Performance Tips

1. Use bfloat16 precision for best performance on TPUs
2. Increase the context window size (`n_ctx`) to make better use of TPU memory
3. Batch multiple requests together for better throughput
4. Use the largest model that fits in TPU memory for best quality

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'torch_xla'**

   Make sure you have installed PyTorch/XLA correctly.

2. **RuntimeError: TPU device not found**

   Make sure you're running on a TPU VM or have access to TPU devices.

3. **Memory errors**

   Try reducing the model size or the number of layers offloaded to TPU.

### Checking TPU Status

You can check the status of your TPU devices with:

```python
import torch_xla.core.xla_model as xm
devices = xm.get_xla_supported_devices()
print(f"Found {len(devices)} TPU devices")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
