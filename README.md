# HelpingAI Inference Server

A professional, production-ready inference server for HelpingAI models with support for multiple hardware platforms (CPU, GPU, TPU) and model formats.

## Recent Updates

### Native Message Format Support

The server now uses the native role (message) feature of transformers and llama-cpp-python:

1. Removed the `DEFAULT_TEMPLATE` and updated the code to use native message formats
2. Updated the `format_prompt` function to return both a formatted prompt and the message objects
3. Enhanced the `generate_text` function to use `create_chat_completion` for llama-cpp models when messages are available
4. Updated the chat completion and completion endpoints to pass messages to the model

### Example Usage with llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="VarunGuptaPy/HelpingAI-3-Q4_K_M-GGUF",
    filename="helpingai-3-q4_k_m.gguf",
)

response = llm.create_chat_completion(
    messages = [
        {
            "role": "system",
            "content": "You are HelpingAI, an emotionally intelligent AI assistant."
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
)
```

## Features

- **Multi-Hardware Support**:
  - Automatic device detection (CPU, GPU, TPU, Apple Silicon)
  - Run on any available hardware with zero configuration
  - Graceful fallbacks when requested hardware is unavailable
  - Full support for Apple Silicon (M1/M2/M3) via MPS backend
- **Model Format Flexibility**:
  - Use fp16 models (default and recommended)
  - Support for GGUF models when needed (especially for TPU compatibility)
  - Automatic GGUF model download from Hugging Face with progress bar
  - Compatible with multiple versions of llama-cpp-python
- **GGUF Metadata Extraction**:
  - Automatically reads metadata from GGUF files
  - Extracts and uses chat templates from GGUF models
  - Provides model architecture and context length information
- **ChatTemplate Support**:
  - Automatically uses model-specific chat templates when available
  - Extracts chat templates from GGUF files
  - Falls back to architecture-specific templates when needed
- **Quantization Options**: Support for 4-bit and 8-bit quantization
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API clients
- **Streaming Support**: Real-time text generation with streaming responses
- **Security**: Optional API key authentication

## Installation

### Basic Dependencies

```bash
pip install -U torch transformers fastapi uvicorn pydantic
```

### Optional Dependencies

For GGUF model support:
```bash
pip install -U llama-cpp-python
```

For TPU support:
```bash
pip install -U torch_xla
```

For progress bar during model downloads:
```bash
pip install -U tqdm
```

## Usage

### Basic Usage (Auto-detect best available device)

```bash
python test.py --model HelpingAI/HelpingAI-15B
```

### Specific Hardware Usage

```bash
# Force CPU usage
python test.py --model HelpingAI/HelpingAI-15B --device cpu

# Force GPU usage
python test.py --model HelpingAI/HelpingAI-15B --device cuda

# Force TPU usage
python test.py --model HelpingAI/HelpingAI-15B --device xla --use-tpu --tpu-cores 8

# Force Apple Silicon (MPS) usage
python test.py --model HelpingAI/HelpingAI-15B --device mps
```

### Using GGUF Models (for TPU or other hardware)

```bash
# Using a local GGUF file
python test.py --enable-gguf --gguf-path path/to/helpingai-15b-q4_k_m.gguf

# Auto-downloading GGUF from Hugging Face
python test.py --model HelpingAI/HelpingAI-15B --enable-gguf --download-gguf

# Specifying a particular GGUF file to download
python test.py --model HelpingAI/HelpingAI-15B --enable-gguf --download-gguf --gguf-filename model-q4_k_m.gguf

# Example using Qwen1.5-0.5B-vortex GGUF model on CPU
python test.py --enable-gguf --download-gguf --model mradermacher/Qwen1.5-0.5B-vortex-GGUF --gguf-filename Qwen1.5-0.5B-vortex.IQ4_XS.gguf --device cpu
```

#### GGUF Model Features

##### API Compatibility

The server is compatible with all versions of the llama-cpp-python API. It will automatically detect and use the most appropriate method for your installed version:

1. `__call__` method (recommended in latest API)
2. `create_completion` method (newer versions)
3. `generate` method with `n_predict` parameter (older versions)
4. `completion` method (fallback option)

This ensures maximum compatibility across different llama-cpp-python versions.

##### Token Counting

The server properly counts tokens for GGUF models, providing accurate usage statistics in the API response. It uses a cascading approach to find the best token counting method:

1. `tokenize` method (newer versions)
2. `token_count` method (some versions)
3. `encode` method (some versions)
4. Fallback to a word-based estimate if no method is available

##### Metadata Extraction

The server automatically extracts metadata from GGUF files, including:

1. Model architecture (Llama, Mistral, Qwen, etc.)
2. Model name and organization
3. Context length
4. Chat template

##### Chat Template Support

The server automatically extracts and uses chat templates from GGUF files. If a chat template is found in the GGUF metadata, it will be used for formatting prompts. If no template is found, the server will try to infer an appropriate template based on the model architecture.

##### Hardware Support

GGUF models can run on multiple hardware platforms:

1. CUDA GPU (via n_gpu_layers parameter)
2. CPU (multi-threaded)
3. Apple Silicon (via Metal/MPS)
4. TPU (experimental)

### Quantized Models for Better Performance

```bash
# 8-bit quantization
python test.py --model HelpingAI/HelpingAI-15B --load-8bit

# 4-bit quantization
python test.py --model HelpingAI/HelpingAI-15B --load-4bit
```

### Adding API Key Security

```bash
python test.py --model HelpingAI/HelpingAI-15B --api-keys "key1,key2,key3"
```

## API Endpoints

The server implements OpenAI-compatible endpoints for easy integration:

### Completions API

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "I feel really happy today because",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Chat Completions API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are HelpingAI, an emotionally intelligent assistant."},
      {"role": "user", "content": "I feel really happy today because I just got a promotion!"}
    ],
    "max_tokens": 150
  }'
```

## Advanced Configuration

### Command Line Options

```
--model              Path to HuggingFace model or local model directory
--model-revision     Specific model revision to load
--tokenizer          Path to tokenizer (defaults to model path)
--tokenizer-revision Specific tokenizer revision to load
--host               Host to bind the server to (default: 0.0.0.0)
--port               Port to bind the server to (default: 8000)
--device             Device to load the model on (auto, cuda, cpu, mps, xla)
--device-map         Device map for model distribution (default: auto)
--dtype              Data type for model weights (float16, float32, bfloat16)
--load-8bit          Load model in 8-bit precision
--load-4bit          Load model in 4-bit precision
--use-tpu            Enable TPU support (requires torch_xla)
--tpu-cores          Number of TPU cores to use (default: 8)
--api-keys           Comma-separated list of valid API keys
--max-concurrent     Maximum number of concurrent requests (default: 10)
--max-queue          Maximum queue size for pending requests (default: 100)
--timeout            Timeout for requests in seconds (default: 60)
--enable-gguf        Enable GGUF model support (requires llama-cpp-python)
--gguf-path          Path to GGUF model file
--download-gguf      Download GGUF model from Hugging Face (if available)
--gguf-filename      Specific GGUF filename to download (e.g., 'model-q4_k_m.gguf')
--num-gpu-layers     Number of GPU layers for GGUF models (-1 means all)
```

## Hardware Recommendations

- **Auto-detect**: For most users, use the default auto-detection for the best experience
- **GPU**: For best performance, use NVIDIA GPUs with at least 8GB VRAM
- **CPU**: For CPU-only deployment, consider using quantized models (4-bit or 8-bit)
- **TPU**: When using Google Cloud TPUs, use bfloat16 precision for optimal performance
- **Apple Silicon**: For Mac users with M1/M2/M3 chips, the MPS backend provides GPU acceleration

## Model Format Recommendations

- **fp16**: Default and recommended for most hardware (GPU, CPU)
- **GGUF**: Use when needed for specific hardware compatibility or when memory optimization is critical
