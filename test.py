"""
HelpingAI Inference Server
A professional, production-ready inference server for HelpingAI models.

Created for commercial deployment of HelpingAI models.
"""

import os
import sys
import time
import logging
import argparse
import json
import uuid
import requests
import shutil
import tarfile
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import threading
import queue
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Add new imports for better hardware detection
try:
    import py_cpuinfo as cpuinfo # type: ignore
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False

# Check for bitsandbytes availability
try:
    import bitsandbytes as bnb # type: ignore
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

# Check for XLA/TPU availability
try:
    import torch_xla # type: ignore
    import torch_xla.core.xla_model as xm # type: ignore
    # Verify TPU is actually available by trying to get devices
    try:
        _ = xm.xla_device()
        XLA_AVAILABLE = True
    except Exception as e:
        print(f"PyTorch/XLA imported but TPU not available: {e}")
        XLA_AVAILABLE = False
except ImportError:
    XLA_AVAILABLE = False

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("helpingai-server")

# Import accelerate for model initialization
try:
    from accelerate import init_empty_weights
except ImportError:
    logger.warning("accelerate not installed, some features may not work")
    logger.warning("To install: pip install accelerate")

    # Provide a fallback implementation
    from contextlib import contextmanager
    @contextmanager
    def init_empty_weights():
        yield

# API key header for authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Server version
SERVER_VERSION = "1.0.0"

# Model cache
MODEL_CACHE = {}
TOKENIZER_CACHE = {}

# We'll use native chat templates instead of a default template

# Custom stopping criteria class
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# Request models
class CompletionRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = "You are HelpingAI, an emotionally intelligent AI assistant."
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    user_id: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    # Required parameters
    messages: List[ChatMessage]

    # OpenAI compatibility parameters
    model: Optional[str] = None  # Model to use (will override the server's default model)
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = 512
    n: Optional[int] = 1  # Number of completions to generate
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, str]] = None  # For JSON mode: {"type": "json_object"}
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40  # Not in OpenAI API but useful for GGUF models
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, str]]] = None
    user: Optional[str] = None  # OpenAI uses 'user' instead of 'user_id'

    # Legacy parameters (for backward compatibility)
    system_prompt: Optional[str] = None  # Will be used if no system message in messages
    user_id: Optional[str] = None  # Legacy parameter, use 'user' instead

class ModelsRequest(BaseModel):
    pass

# Response models
class CompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for this completion")
    object: str = Field("text_completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation time")
    model: str = Field(..., description="Model used for completion")
    choices: List[Dict[str, Any]] = Field(..., description="Completion choices")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for this chat completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation time")
    model: str = Field(..., description="Model used for completion")
    choices: List[Dict[str, Any]] = Field(..., description="Completion choices")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint for this completion")

class ModelsResponse(BaseModel):
    object: str = Field("list", description="Object type")
    data: List[Dict[str, Any]] = Field(..., description="List of available models")

# Helper function to download GGUF models from Hugging Face
def download_gguf_from_hf(model_id: str, filename: str = None, revision: str = None) -> str:
    """
    Download a GGUF model file from Hugging Face Hub.

    Args:
        model_id: The Hugging Face model ID (e.g., 'HelpingAI/HelpingAI-15B')
        filename: The specific GGUF filename to download (e.g., 'model-q4_k_m.gguf')
                  If None, will try to find a GGUF file in the model files
        revision: The specific model revision to download

    Returns:
        Path to the downloaded GGUF file
    """
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Create a directory for this specific model
    model_dir = models_dir / model_id.replace("/", "_")
    model_dir.mkdir(exist_ok=True)

    # Construct the Hugging Face API URL to get model info
    api_url = f"https://huggingface.co/api/models/{model_id}"
    if revision:
        api_url += f"/revision/{revision}"

    try:
        # Get model info from Hugging Face API
        response = requests.get(api_url)
        response.raise_for_status()
        model_info = response.json()

        # If no specific filename is provided, try to find a GGUF file
        if not filename:
            siblings = model_info.get("siblings", [])
            gguf_files = [s["rfilename"] for s in siblings if s["rfilename"].endswith(".gguf")]

            if not gguf_files:
                raise ValueError(f"No GGUF files found in model {model_id}")

            # Use the first GGUF file found (or could implement logic to choose the best one)
            filename = gguf_files[0]
            logger.info(f"Found GGUF file: {filename}")

        # Construct the download URL
        download_url = f"https://huggingface.co/{model_id}/resolve/"
        if revision:
            download_url += f"{revision}/"
        else:
            download_url += "main/"
        download_url += filename

        # Path where the file will be saved
        output_path = model_dir / filename

        # Download the file if it doesn't exist already
        if not output_path.exists():
            logger.info(f"Downloading GGUF model from {download_url}")
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()

                # Get file size for progress bar
                file_size = int(r.headers.get('content-length', 0))

                # Create progress bar if tqdm is available
                if TQDM_AVAILABLE and file_size > 0:
                    progress_bar = tqdm(
                        total=file_size,
                        unit='iB',
                        unit_scale=True,
                        desc=f"Downloading {filename}"
                    )

                    with open(output_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                progress_bar.update(len(chunk))
                    progress_bar.close()
                else:
                    # Fallback if tqdm is not available
                    with open(output_path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)

            logger.info(f"Downloaded GGUF model to {output_path}")
        else:
            logger.info(f"Using cached GGUF model at {output_path}")

        return str(output_path)

    except Exception as e:
        logger.error(f"Error downloading GGUF model: {e}")
        raise

# Server configuration
@dataclass
class ServerConfig:
    model_name_or_path: str
    model_revision: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    device_map: str = "auto"
    dtype: str = "float16"
    load_8bit: bool = False
    load_4bit: bool = False
    api_keys: List[str] = field(default_factory=list)
    max_concurrent_requests: int = 10
    max_queue_size: int = 100
    timeout: int = 60
    num_gpu_layers: int = -1
    use_cache: bool = True
    enable_gguf: bool = False
    gguf_path: Optional[str] = None
    gguf_filename: Optional[str] = None
    download_gguf: bool = False
    use_tpu: bool = False
    tpu_cores: int = 8
    tpu_layers: int = -1
    tpu_bf16: bool = True
    tpu_memory_limit: str = "1GB"

    def __post_init__(self):
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path

        # Enhanced hardware detection and configuration
        if self.device == "auto":
            logger.info("Auto-detecting optimal hardware configuration...")

            # Check for CUDA first (NVIDIA GPU)
            if torch.cuda.is_available():
                cuda_devices = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    cuda_devices.append({
                        "index": i,
                        "name": props.name,
                        "total_memory": props.total_memory,
                        "compute_capability": f"{props.major}.{props.minor}"
                    })

                # Log discovered CUDA devices
                logger.info(f"Found {len(cuda_devices)} CUDA device(s):")
                for dev in cuda_devices:
                    logger.info(f"  [{dev['index']}] {dev['name']} - {dev['total_memory'] / (1024**3):.2f} GB - Compute {dev['compute_capability']}")

                self.device = "cuda"

                # Auto-configure quantization based on available VRAM
                best_gpu = max(cuda_devices, key=lambda x: x["total_memory"])
                available_vram = best_gpu["total_memory"] / (1024**3)  # Convert to GB

                logger.info(f"Auto-configuring for {available_vram:.2f} GB VRAM")

                # Set quantization automatically based on VRAM
                if available_vram < 8:
                    logger.info("Low VRAM detected (< 8GB), enabling 4-bit quantization")
                    self.load_4bit = True
                elif available_vram < 16:
                    logger.info("Medium VRAM detected (8-16GB), enabling 8-bit quantization")
                    self.load_8bit = True
                else:
                    logger.info(f"High VRAM detected ({available_vram:.2f}GB), using FP16")

                # Set appropriate device map for multi-GPU
                if len(cuda_devices) > 1:
                    logger.info(f"Multiple GPUs detected, using automatic device map")
                    self.device_map = "auto"
                else:
                    # For single GPU, may be better to specify the device directly
                    self.device_map = {"": 0}

            # Check for MPS (Apple Silicon GPU)
            elif hasattr(torch, 'has_mps') and torch.backends.mps.is_available():
                logger.info("Apple Silicon GPU (MPS) detected")
                self.device = "mps"

                # Apple Silicon specific optimizations
                # M1/M2/M3 chip detection
                import platform
                import subprocess

                try:
                    # Get detailed Apple Silicon info
                    sysctl_output = subprocess.check_output(['sysctl', 'hw.model']).decode('utf-8').strip()
                    model_name = sysctl_output.split(':')[1].strip() if ':' in sysctl_output else "Unknown"

                    # Estimate memory from system info
                    memory_output = subprocess.check_output(['sysctl', 'hw.memsize']).decode('utf-8').strip()
                    system_memory = int(memory_output.split(':')[1].strip()) / (1024**3) if ':' in memory_output else 0

                    logger.info(f"Detected Apple Silicon device: {model_name} with ~{system_memory:.2f}GB system memory")

                    # Set appropriate settings based on device type
                    if "M3" in model_name or "M2" in model_name:
                        logger.info("High-performance Apple Silicon detected, using optimized settings")
                        self.dtype = "bfloat16" if hasattr(torch, "bfloat16") else "float16"
                    else:
                        logger.info("Using float16 precision for Apple Silicon")
                        self.dtype = "float16"

                    # Apple Silicon benefits from 4-bit quantization for larger models
                    if "gguf" in self.model_name_or_path.lower() or self.enable_gguf:
                        logger.info("Using GGUF with Metal acceleration for Apple Silicon")
                        self.enable_gguf = True
                        self.num_gpu_layers = -1  # Use all layers on MPS
                except Exception as e:
                    logger.warning(f"Error detecting Apple Silicon details: {e}")

            # Check for TPU
            elif self.use_tpu or any("TPU" in env for env in os.environ if isinstance(env, str)):
                if XLA_AVAILABLE:
                    try:
                        logger.info("TPU hardware detected")
                        self.use_tpu = True
                        self.device = "xla"

                        # Set TPU environment variables
                        os.environ["PJRT_DEVICE"] = "TPU"
                        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

                        # Enable appropriate TPU optimizations
                        self.tpu_bf16 = True  # BF16 is faster on TPU

                        # Set TPU-specific parameters
                        if self.tpu_layers <= 0:
                            self.tpu_layers = -1  # Use all layers

                        logger.info(f"TPU configuration: using {self.tpu_layers} layers on TPU")
                        logger.info(f"TPU precision: {'BF16' if self.tpu_bf16 else 'default'}")

                    except Exception as e:
                        logger.warning(f"TPU environment detected but error initializing: {e}")
                        self.use_tpu = False
                        self.device = "cpu"
                else:
                    logger.warning("TPU environment detected but torch_xla not available")
                    logger.warning("Install with: pip install torch_xla")
                    self.use_tpu = False
                    self.device = "cpu"

            # Fallback to CPU with optimal threading
            else:
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                logger.info(f"No GPU/TPU detected. Using CPU with {cpu_count} cores")
                self.device = "cpu"

                # Set number of threads for CPU inference
                if hasattr(torch, 'set_num_threads'):
                    torch.set_num_threads(cpu_count)
                    logger.info(f"Set PyTorch to use {cpu_count} threads")

                # For CPU, smaller models or GGUF is recommended
                logger.info("For CPU inference, GGUF models are recommended for better performance")
                if not self.enable_gguf and "gguf" not in self.model_name_or_path.lower():
                    logger.info("Automatically enabling GGUF for CPU inference")
                    self.enable_gguf = True
                    self.download_gguf = True

        # Specific device configuration (when not using auto)
        # TPU Configuration
        if self.use_tpu:
            try:
                import torch_xla.core.xla_model as xm # type: ignore
                import torch_xla.distributed.xla_backend # type: ignore
                import torch_xla.distributed.parallel_loader as pl # type: ignore
                import torch_xla.utils.utils as xu # type: ignore

                logger.info("TPU support enabled")
                self.device = "xla"

                # Get TPU device count
                tpu_devices = xm.get_xla_supported_devices()
                logger.info(f"Found {len(tpu_devices)} TPU devices")

                # Set TPU-specific environment variables
                os.environ['XLA_USE_BF16'] = '1' if self.tpu_bf16 else '0'  # Enable bfloat16 for better performance

                # Convert memory limit string to bytes (e.g., "1GB" -> 1000000000)
                memory_limit = self.tpu_memory_limit.upper()
                if memory_limit.endswith('GB'):
                    memory_bytes = int(float(memory_limit[:-2]) * 1000000000)
                elif memory_limit.endswith('MB'):
                    memory_bytes = int(float(memory_limit[:-2]) * 1000000)
                else:
                    memory_bytes = 1000000000  # Default to 1GB

                os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = str(memory_bytes)

                # Set TPU-specific parameters for llama.cpp
                os.environ['GGML_TPU_ENABLE'] = '1'
                os.environ['GGML_TPU_LAYERS'] = str(self.tpu_layers if self.tpu_layers != 0 else self.num_gpu_layers)

                # Log TPU configuration
                logger.info(f"TPU configuration: BF16={self.tpu_bf16}, Memory limit={self.tpu_memory_limit}, Layers={self.tpu_layers if self.tpu_layers != 0 else self.num_gpu_layers}")
            except ImportError:
                logger.warning("TPU support requested but torch_xla not available, falling back to CPU/GPU")
                self.use_tpu = False
                # Continue with CUDA/CPU/MPS check

        # Check for MPS (Apple Silicon GPU)
        if not self.use_tpu and self.device == "mps":
            if hasattr(torch, 'has_mps') and torch.has_mps and torch.backends.mps.is_available():
                logger.info("MPS (Apple Silicon GPU) is available and will be used")

                # Set Metal environment variables for optimal performance with GGUF
                if self.enable_gguf:
                    os.environ["GGML_METAL_ENABLE"] = "1"
                    os.environ["GGML_METAL_NDEBUGGROUPS"] = "32"  # Performance optimization
                    logger.info("Enabled Metal optimizations for GGUF models")
            else:
                logger.warning("MPS (Apple Silicon GPU) requested but not available, falling back to CPU")
                self.device = "cpu"

        # Check for CUDA
        elif not self.use_tpu and self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Log final device configuration
        logger.info(f"Final device configuration: {self.device}")
        if self.device == "cuda":
            logger.info(f"CUDA configuration: 4-bit={self.load_4bit}, 8-bit={self.load_8bit}, fp16={(self.dtype == 'float16')}")

        # Log GGUF configuration
        if self.enable_gguf:
            logger.info(f"GGUF enabled: using {'downloaded' if self.download_gguf else 'local'} model, GPU layers={self.num_gpu_layers}")

# Task queue for request handling
class TaskQueue:
    def __init__(self, max_concurrent: int, max_queue_size: int):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                task, args, kwargs = self.queue.get(timeout=0.1)
                with self.semaphore:
                    task(*args, **kwargs)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Task execution error: {e}")

    def add_task(self, task, *args, **kwargs):
        try:
            self.queue.put((task, args, kwargs), block=False)
            return True
        except queue.Full:
            return False

    def shutdown(self):
        self._stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

# Helper functions
def create_app(config: ServerConfig):
    app = FastAPI(
        title="HelpingAI Inference Server",
        description=f"Commercial Inference Server for HelpingAI models - v{SERVER_VERSION}",
        version=SERVER_VERSION
    )

    # Setup middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize the task queue
    task_queue = TaskQueue(config.max_concurrent_requests, config.max_queue_size)

    # Store config
    app.state.config = config
    app.state.task_queue = task_queue

    return app

def load_model(config: ServerConfig):
    model_id = config.model_name_or_path

    if model_id in MODEL_CACHE:
        logger.info(f"Using cached model: {model_id}")
        return MODEL_CACHE[model_id], TOKENIZER_CACHE[model_id]

    logger.info(f"Loading model: {model_id}")
    start_time = time.time()

    # First load the tokenizer as it's needed for both GGUF and transformers paths
    try:
        logger.info(f"Loading tokenizer from {config.tokenizer_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            revision=config.tokenizer_revision,
            use_fast=True,
            trust_remote_code=True  # Allow custom tokenizer code
        )

        # Add special tokens if needed
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Last resort for models without eos token
                tokenizer.pad_token = tokenizer.unk_token or "<|padding|>"

        # Report on tokenizer capabilities
        has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
        logger.info(f"Tokenizer loaded successfully from {config.tokenizer_name_or_path}. Chat template: {'Available' if has_chat_template else 'Not available'}")
    except Exception as e:
        logger.warning(f"Error loading tokenizer: {e}")
        tokenizer = None
        logger.info("Will attempt to proceed without tokenizer (GGUF models can work without it)")

    # Auto-detect model format if not explicitly specified
    is_gguf_model = config.enable_gguf
    if not is_gguf_model:
        # Auto-detect if this is likely a GGUF model
        is_gguf_model = (
            "gguf" in model_id.lower() or
            (config.gguf_filename and config.gguf_filename.lower().endswith(".gguf"))
        )
        if is_gguf_model:
            logger.info(f"Auto-detected GGUF model format from model name: {model_id}")
            config.enable_gguf = True

    # Handle GGUF model path - download from HF if requested
    gguf_path = config.gguf_path
    if config.download_gguf or (config.enable_gguf and not gguf_path):
        try:
            logger.info(f"Attempting to download GGUF model for {model_id}")
            gguf_path = download_gguf_from_hf(
                model_id=model_id,
                filename=config.gguf_filename,
                revision=config.model_revision
            )
            logger.info(f"Successfully downloaded GGUF model to {gguf_path}")
        except Exception as e:
            logger.error(f"Failed to download GGUF model: {e}")
            if not config.enable_gguf:
                logger.info("Falling back to standard transformers model")
                gguf_path = None
            else:
                raise

    # GGUF path - use if explicitly enabled or if TPU is used and GGUF path is provided
    if (config.enable_gguf and gguf_path) or (config.use_tpu and gguf_path):
        # GGUF loading logic requires llama-cpp-python
        try:
            from llama_cpp import Llama
            logger.info(f"Loading GGUF model from {gguf_path}")

            # Configure GGUF model based on available hardware
            gguf_kwargs = {
                "model_path": gguf_path,
                "n_ctx": 4096,  # Context size
                "verbose": False
            }

            # Configure GPU usage for GGUF
            if torch.cuda.is_available() and config.device == "cuda":
                # Configure for CUDA
                gguf_kwargs["n_gpu_layers"] = config.num_gpu_layers
                gguf_kwargs["use_mlock"] = True  # Helps with GPU memory management

                # Get CUDA device properties for better configuration
                props = torch.cuda.get_device_properties(0)
                cuda_compute_capability = f"{props.major}.{props.minor}"

                # Set GGML CUDA parameters based on GPU capabilities
                os.environ["GGML_CUDA_CAPABILITY"] = cuda_compute_capability

                # For newer GPUs with Tensor Cores, enable them
                if float(cuda_compute_capability) >= 7.0:
                    os.environ["GGML_CUDA_TENSOR_CORES"] = "1"

                logger.info(f"Configuring GGUF model for CUDA GPU with compute capability {cuda_compute_capability}")
                logger.info(f"Using {config.num_gpu_layers} layers on GPU")

            elif config.device == "mps":
                # Configure for Apple Silicon (Metal)
                logger.info("Configuring GGUF model for Apple Silicon GPU (Metal)")

                # Enable Metal acceleration
                os.environ["GGML_METAL_ENABLE"] = "1"
                gguf_kwargs["use_mlock"] = True
                gguf_kwargs["n_gpu_layers"] = config.num_gpu_layers if config.num_gpu_layers > 0 else -1

                # Performance optimizations for Metal
                os.environ["GGML_METAL_NDEBUGGROUPS"] = "32"  # Performance optimization

                logger.info(f"Metal acceleration enabled with {config.num_gpu_layers if config.num_gpu_layers > 0 else 'all'} layers")

            elif config.use_tpu:
                # TPU-specific configuration for GGUF
                logger.info("Configuring GGUF model for TPU")
                gguf_kwargs["n_gpu_layers"] = config.tpu_layers if config.tpu_layers > 0 else -1
                gguf_kwargs["use_mlock"] = False  # mlock not needed for TPU

                # Set environment variables for TPU support
                os.environ["PJRT_DEVICE"] = "TPU"
                os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

                # Use BF16 precision if requested (better for TPU)
                if config.tpu_bf16:
                    os.environ["GGML_TPU_BF16"] = "1"

                logger.info(f"TPU configuration: using {config.tpu_layers if config.tpu_layers > 0 else 'all'} layers on TPU")
                logger.info(f"TPU precision: {'BF16' if config.tpu_bf16 else 'default'}")
            else:
                # CPU configuration with performance optimizations
                import multiprocessing
                num_threads = os.cpu_count() or multiprocessing.cpu_count()
                gguf_kwargs["n_threads"] = num_threads

                # Check for AVX/AVX2/AVX512 support for better performance
                if CPUINFO_AVAILABLE:
                    try:
                        import cpuinfo # type: ignore
                        cpu_info = cpuinfo.get_cpu_info()
                        cpu_flags = cpu_info.get('flags', [])

                        # Log CPU capabilities
                        cpu_features = []
                        if 'avx512f' in cpu_flags:
                            cpu_features.append('AVX-512')
                            os.environ["GGML_AVX512"] = "1"
                        if 'avx2' in cpu_flags:
                            cpu_features.append('AVX2')
                            os.environ["GGML_AVX2"] = "1"
                        elif 'avx' in cpu_flags:
                            cpu_features.append('AVX')
                            os.environ["GGML_AVX"] = "1"

                        if cpu_features:
                            logger.info(f"CPU supports {', '.join(cpu_features)} instructions")

                    except Exception as e:
                        logger.warning(f"Could not detect CPU features: {e}")
                    else:
                        logger.info("py-cpuinfo not installed, skipping CPU feature detection")
                        logger.info("Install with: pip install py-cpuinfo for better CPU performance")

                logger.info(f"Configuring GGUF model for CPU with {num_threads} threads")

            # Add GGUF-specific parameters for optimal performance

            # Set batch size based on device
            if config.device == "cuda" or config.device == "mps":
                gguf_kwargs["n_batch"] = 512  # Larger batch size for GPU
            else:
                gguf_kwargs["n_batch"] = 256  # Smaller batch size for CPU

            # Load the GGUF model
            logger.info(f"Creating GGUF model instance with parameters: {gguf_kwargs}")
            model = Llama(**gguf_kwargs)
            logger.info(f"GGUF model loaded successfully in {time.time() - start_time:.2f} seconds")

        except ImportError as e:
            logger.error(f"llama-cpp-python not installed or missing dependencies: {e}")
            if config.use_tpu or config.device == "mps" or gguf_path:
                logger.error("Attempting to fall back to transformers with fp16...")
                config.enable_gguf = False
            else:
                raise

    # Transformers path if not using GGUF
    if not config.enable_gguf or (config.enable_gguf and not gguf_path):
        logger.info("Loading model with Transformers")

        # Set up device-specific configurations
        if config.use_tpu:
            try:
                import torch_xla.core.xla_model as xm # type: ignore
                device = xm.xla_device()
                model_kwargs = {
                    "device_map": None,  # Don't use device_map with TPU
                    "torch_dtype": torch.bfloat16,  # bfloat16 is preferred for TPU
                }
                logger.info("Using TPU configuration for model loading")
            except ImportError:
                logger.warning("TPU support requested but torch_xla not available, using CPU/GPU configuration")
                model_kwargs = {
                    "device_map": config.device_map,
                    "torch_dtype": torch.float16 if config.dtype == "float16" else torch.float32,
                    "use_cache": config.use_cache,
                }
        elif config.device == "mps":
            # MPS (Apple Silicon) configuration
            model_kwargs = {
                "device_map": None,  # We'll move the model to MPS after loading
                "torch_dtype": torch.float16 if config.dtype == "float16" else torch.float32,
                "use_cache": config.use_cache,
            }
            logger.info("Using MPS (Apple Silicon GPU) configuration for model loading")
        else:
            # Standard CPU/GPU configuration
            model_kwargs = {
                "device_map": config.device_map,
                "torch_dtype": torch.float16 if config.dtype == "float16" else torch.float32,
                "use_cache": config.use_cache,
            }

        # Add quantization options if specified
        if config.load_8bit:
            logger.info("Loading model in 8-bit quantization")
            if BNB_AVAILABLE:
                model_kwargs["load_in_8bit"] = True
                model_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
                logger.info("Using bitsandbytes for 8-bit quantization")
            else:
                logger.warning("8-bit quantization requested but bitsandbytes not available")
                logger.warning("To enable 8-bit: pip install bitsandbytes")
                config.load_8bit = False

        if config.load_4bit:
            logger.info("Loading model in 4-bit quantization")
            if BNB_AVAILABLE:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                model_kwargs["bnb_4bit_quant_type"] = "nf4"
                model_kwargs["bnb_4bit_use_double_quant"] = True
                logger.info("Using bitsandbytes for 4-bit quantization")
            else:
                logger.warning("4-bit quantization requested but bitsandbytes not available")
                logger.warning("To enable 4-bit: pip install bitsandbytes")
                config.load_4bit = False

        # Add trust_remote_code for models that need it
        model_kwargs["trust_remote_code"] = True

        # Add low_cpu_mem_usage for better memory management during loading
        model_kwargs["low_cpu_mem_usage"] = True

        # Load the model
        try:
            logger.info(f"Loading model with Transformers using parameters: {model_kwargs}")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                revision=config.model_revision,
                **model_kwargs
            )

            # Move model to MPS if using MPS
            if config.device == "mps" and hasattr(torch, 'has_mps') and torch.has_mps and torch.backends.mps.is_available():
                logger.info("Moving model to MPS device")
                model = model.to(torch.device("mps"))
                logger.info("Model moved to MPS device")
            # Move model to TPU if using TPU
            elif config.use_tpu and 'device' in locals():
                logger.info("Moving model to TPU device")
                model = model.to(device)

            logger.info(f"Model loaded successfully on {config.device} in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading model with transformers: {e}")

            # Try to provide helpful error messages based on common issues
            if "CUDA out of memory" in str(e):
                logger.error("GPU out of memory error. Try using --load-8bit or --load-4bit to reduce memory usage.")
            elif "Cannot handle type" in str(e) and config.device == "mps":
                logger.error("MPS compatibility issue. Apple Silicon may not support all model operations.")
                logger.error("Try using --device cpu or using a GGUF model with --enable-gguf.")

            raise

    # Cache the model and tokenizer for future use
    MODEL_CACHE[model_id] = model
    TOKENIZER_CACHE[model_id] = tokenizer

    logger.info("Model and tokenizer successfully loaded and cached")
    return model, tokenizer

def verify_api_key(api_key: str = Depends(API_KEY_HEADER), config: ServerConfig = None):
    if not config.api_keys:
        return True  # No API keys configured, allow all
    if api_key not in config.api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

def format_prompt(messages, system_prompt, tokenizer):
    """Format messages into the appropriate prompt format using ChatTemplate when available."""
    if not messages:
        return system_prompt, []

    # Format messages for the tokenizer's chat template
    formatted_messages = []

    # Check if there's already a system message in the messages array
    system_messages = [msg for msg in messages if msg.role.lower() == "system"]
    has_system_message = len(system_messages) > 0

    # Add system message if provided and not already in messages
    if system_prompt and not has_system_message:
        formatted_messages.append({"role": "system", "content": system_prompt})

    # Add all messages
    for msg in messages:
        formatted_messages.append({"role": msg.role.lower(), "content": msg.content})

    # Log the formatted messages for debugging
    logger.info(f"Formatted messages: {formatted_messages}")

    # Check if tokenizer has chat_template
    if tokenizer and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        # Apply the model's chat template
        try:
            prompt = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.info("Using model's chat template for prompt formatting")
            return prompt, formatted_messages
        except Exception as e:
            logger.warning(f"Error applying chat template: {e}. Will try to use native message format.")

    # For models without a chat template, we'll return the formatted messages
    # to be used with create_chat_completion for llama-cpp models
    # or we'll create a simple concatenated prompt for other models

    # For non-llama-cpp models that don't have a chat template, create a simple prompt
    last_user_message = None
    for msg in reversed(messages):
        if msg.role.lower() == "user":
            last_user_message = msg.content
            break

    if last_user_message is None:
        last_user_message = ""

    # Create a simple concatenated prompt as fallback
    simple_prompt = f"System: {system_prompt}\n\nUser: {last_user_message}\n\nAssistant:"

    return simple_prompt, formatted_messages

def count_tokens_gguf(model, text):
    """Count tokens for GGUF models using llama-cpp-python."""
    try:
        # Try different methods to count tokens based on the API version
        if hasattr(model, "tokenize"):
            # Newer versions have a tokenize method
            tokens = model.tokenize(text.encode('utf-8'))
            return len(tokens)
        elif hasattr(model, "token_count"):
            # Some versions have a token_count method
            return model.token_count(text)
        elif hasattr(model, "encode"):
            # Some versions have an encode method
            tokens = model.encode(text)
            return len(tokens)
        else:
            # Rough estimate based on words (fallback)
            logger.warning("No token counting method available for GGUF model, using rough estimate")
            return len(text.split()) * 1.3  # Rough estimate
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}. Using rough estimate.")
        return len(text.split()) * 1.3  # Rough estimate

def generate_text(model, tokenizer, prompt, params, stream=False, messages=None):
    """Generate text using the model."""

    # Check if using GGUF model (llama-cpp-python)
    if isinstance(model, object) and tokenizer is None:
        # According to the latest llama-cpp-python documentation
        # The recommended way to generate text is to use the __call__ method
        # or create_completion method which provides an OpenAI-compatible interface
        logger.info(f"Generating with GGUF model using prompt: {prompt[:50]}...")

        try:
            # Check if we have messages and can use create_chat_completion
            if messages and hasattr(model, "create_chat_completion"):
                logger.info("Using model.create_chat_completion method with native message format")
                # Prepare parameters for chat completion
                # Convert messages to the format expected by llama_cpp
                # This ensures proper role-based messaging
                llama_cpp_messages = []

                # Log the messages we're processing
                logger.info(f"Processing messages for llama_cpp: {messages}")

                for msg in messages:
                    llama_cpp_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

                # Log the final messages being sent to the model
                logger.info(f"Final llama_cpp messages: {llama_cpp_messages}")

                # Prepare OpenAI-compatible parameters for llama_cpp
                chat_params = {
                    "messages": llama_cpp_messages,
                    "max_tokens": params["max_tokens"],
                    "temperature": params["temperature"],
                    "top_p": params["top_p"],
                    "top_k": params["top_k"] if "top_k" in params else 40,
                    "stream": stream
                }

                # Add stop tokens if provided
                if params["stop"] and len(params["stop"]) > 0:
                    chat_params["stop"] = params["stop"]

                # Add frequency_penalty if provided
                if "frequency_penalty" in params:
                    chat_params["frequency_penalty"] = params["frequency_penalty"]

                # Add presence_penalty if provided
                if "presence_penalty" in params:
                    chat_params["presence_penalty"] = params["presence_penalty"]

                # Add n (number of completions) if provided
                if "n" in params and params["n"] > 1:
                    chat_params["n_predict"] = params["n"]

                # Add seed if provided
                if "seed" in params:
                    chat_params["seed"] = params["seed"]

                # Add logit_bias if provided
                if "logit_bias" in params and params["logit_bias"]:
                    chat_params["logit_bias"] = params["logit_bias"]

                # Add tools if provided
                if "tools" in params and params["tools"]:
                    chat_params["tools"] = params["tools"]

                    if "tool_choice" in params and params["tool_choice"]:
                        chat_params["tool_choice"] = params["tool_choice"]

                output = model.create_chat_completion(**chat_params)

                if stream:
                    return output  # This is a generator for streaming
                else:
                    # Extract the generated text from the response
                    if isinstance(output, dict) and "choices" in output:
                        return output["choices"][0]["message"]["content"]
                    else:
                        return output.strip()

            # If no messages or create_chat_completion not available, use regular completion
            # Prepare parameters for the model
            completion_params = {
                "prompt": prompt,
                "max_tokens": params["max_tokens"],
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "top_k": params["top_k"],
                "stream": stream
            }

            # Add stop tokens if provided
            if params["stop"] and len(params["stop"]) > 0:
                completion_params["stop"] = params["stop"]

            # Try the __call__ method first (recommended in latest API)
            if hasattr(model, "__call__"):
                logger.info("Using model.__call__ method (recommended API)")
                output = model(**completion_params)

                if stream:
                    return output  # This is a generator for streaming
                else:
                    # Extract the generated text from the response
                    if isinstance(output, dict) and "choices" in output:
                        return output["choices"][0]["text"]
                    else:
                        return output.strip()

            # Try create_completion as a fallback
            elif hasattr(model, "create_completion"):
                logger.info("Using model.create_completion method")
                output = model.create_completion(**completion_params)

                if stream:
                    return output  # This is a generator for streaming
                else:
                    # Extract the generated text from the response
                    if isinstance(output, dict) and "choices" in output:
                        return output["choices"][0]["text"]
                    else:
                        return output.strip()

            # Try the older generate method with n_predict instead of max_tokens
            elif hasattr(model, "generate"):
                logger.info("Using older model.generate method with n_predict")
                # Convert max_tokens to n_predict for older API
                gguf_params = completion_params.copy()
                gguf_params["n_predict"] = gguf_params.pop("max_tokens")

                output = model.generate(**gguf_params)

                if stream:
                    return output  # This is a generator for streaming
                else:
                    # Extract the generated text from the response
                    if isinstance(output, dict) and "choices" in output:
                        return output["choices"][0]["text"]
                    else:
                        return output.strip()

            # Last resort - try the completion method
            elif hasattr(model, "completion"):
                logger.info("Using model.completion method")
                result = model.completion(prompt, max_tokens=params["max_tokens"])
                return result

            else:
                raise ValueError("GGUF model does not have any supported generation methods")

        except Exception as e:
            logger.error(f"Error generating with GGUF model: {e}")
            logger.error("Please check your llama-cpp-python version and model compatibility")
            logger.error("For the latest API, see: https://llama-cpp-python.readthedocs.io/en/latest/")
            raise

    # Using transformers
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Setup stopping criteria
    stop_token_ids = []
    if params["stop"]:
        for stop_str in params["stop"]:
            stop_ids = tokenizer.encode(stop_str, add_special_tokens=False)
            stop_token_ids.extend(stop_ids)

    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)] if stop_token_ids else [])

    # Setup streamer if streaming
    streamer = None
    if stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_config = {
        "max_new_tokens": params["max_tokens"],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "do_sample": params["temperature"] > 0,
        "stopping_criteria": stopping_criteria,
        "use_cache": True,
    }

    if streamer:
        generation_config["streamer"] = streamer

    if stream:
        # Start generation in a separate thread
        thread = threading.Thread(
            target=lambda: model.generate(**inputs, **generation_config)
        )
        thread.start()
        return streamer
    else:
        # Generate directly
        output_ids = model.generate(**inputs, **generation_config)
        return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# API endpoints
def create_routes(app):
    # Check API key middleware
    @app.middleware("http")
    async def check_api_key_middleware(request: Request, call_next):
        config = request.app.state.config

        # Skip API key check for non-API endpoints
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        # Skip API key check if no API keys are configured
        if not config.api_keys:
            return await call_next(request)

        # Check API key
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in config.api_keys:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"}
            )

        return await call_next(request)

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": SERVER_VERSION}

    # Models endpoint
    @app.post("/v1/models")
    async def list_models(request: ModelsRequest):
        config = app.state.config

        models = [{
            "id": config.model_name_or_path,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "HelpingAI",
        }]

        return ModelsResponse(
            object="list",
            data=models
        )

    # Completions endpoint
    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequest, background_tasks: BackgroundTasks):
        config = app.state.config
        model, tokenizer = load_model(config)

        # Define parameters
        params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop": request.stop,
        }

        # Format prompt if needed
        prompt = request.prompt
        if not prompt.strip():
            return HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Create a simple message format for llama-cpp models that support chat completion
        formatted_messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Handle streaming
        if request.stream:
            async def generate_stream():
                # Start generation
                streamer = generate_text(model, tokenizer, prompt, params, stream=True, messages=formatted_messages)

                # Stream the results
                completion_id = f"cmpl-{uuid.uuid4()}"
                created = int(time.time())

                # Send the first chunk
                chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": config.model_name_or_path,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                # Stream the generated text
                collected_text = ""
                for text in streamer:
                    collected_text += text
                    chunk = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": config.model_name_or_path,
                        "choices": [
                            {
                                "index": 0,
                                "text": text,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Send the final chunk
                chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": config.model_name_or_path,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(tokenizer.encode(prompt)) if tokenizer else count_tokens_gguf(model, prompt),
                        "completion_tokens": len(tokenizer.encode(collected_text)) if tokenizer else count_tokens_gguf(model, collected_text),
                        "total_tokens": (len(tokenizer.encode(prompt)) + len(tokenizer.encode(collected_text))) if tokenizer else (count_tokens_gguf(model, prompt) + count_tokens_gguf(model, collected_text)),
                    },
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Non-streaming request
            output = generate_text(model, tokenizer, prompt, params, stream=False, messages=formatted_messages)

            # Calculate token usage
            if tokenizer:
                prompt_tokens = len(tokenizer.encode(prompt))
                completion_tokens = len(tokenizer.encode(output))
            else:
                # Use GGUF token counting for llama-cpp-python models
                prompt_tokens = count_tokens_gguf(model, prompt)
                completion_tokens = count_tokens_gguf(model, output)

            # Format response
            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4()}",
                object="text_completion",
                created=int(time.time()),
                model=config.model_name_or_path,
                choices=[
                    {
                        "index": 0,
                        "text": output,
                        "finish_reason": "stop",
                    }
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )

    # Chat completions endpoint
    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
        config = app.state.config

        # If model is specified in the request, try to load that model instead
        if request.model and request.model != config.model_name_or_path:
            logger.info(f"Client requested model: {request.model}, different from server default: {config.model_name_or_path}")
            try:
                # Create a temporary config with the requested model
                temp_config = ServerConfig(
                    model_name_or_path=request.model,
                    enable_gguf=config.enable_gguf,
                    download_gguf=config.download_gguf,
                    device=config.device
                )
                model, tokenizer = load_model(temp_config)
                logger.info(f"Successfully loaded client-requested model: {request.model}")
            except Exception as e:
                logger.warning(f"Failed to load client-requested model {request.model}: {e}. Falling back to default model.")
                model, tokenizer = load_model(config)
        else:
            model, tokenizer = load_model(config)

        # Define parameters - include all OpenAI compatible parameters
        params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop": request.stop,
            "n": request.n,  # Number of completions to generate
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
        }

        # Add seed if provided for deterministic outputs
        if request.seed is not None:
            params["seed"] = request.seed

        # Handle logit_bias if provided
        if request.logit_bias:
            params["logit_bias"] = request.logit_bias

        # Set user identifier (OpenAI compatibility)
        user = request.user or request.user_id
        if user:
            params["user"] = user

        # Set default system prompt if none is provided in messages or as parameter
        system_prompt = request.system_prompt
        if system_prompt is None:
            # Check if there's a system message in the messages array
            system_messages = [msg for msg in request.messages if msg.role.lower() == "system"]
            if not system_messages:
                # Use default system prompt if no system message is found
                system_prompt = "You are HelpingAI, an emotionally intelligent AI assistant."

        # Format messages into prompt and get formatted messages for native chat
        prompt, formatted_messages = format_prompt(request.messages, system_prompt, tokenizer)

        # Handle JSON mode if requested
        if request.response_format and request.response_format.get("type") == "json_object":
            logger.info("JSON mode requested, adding instruction to generate JSON")
            # Add instruction to generate JSON to the prompt
            if isinstance(prompt, str):
                prompt += "\nYou must respond with a valid JSON object only, with no other text."
            # Also add to the last message if using formatted messages
            if formatted_messages:
                last_user_msg_idx = -1
                for i, msg in enumerate(formatted_messages):
                    if msg["role"] == "user":
                        last_user_msg_idx = i

                if last_user_msg_idx >= 0:
                    formatted_messages[last_user_msg_idx]["content"] += "\nYou must respond with a valid JSON object only, with no other text."
                else:
                    # Add a new system message if no user message found
                    formatted_messages.append({
                        "role": "system",
                        "content": "You must respond with a valid JSON object only, with no other text."
                    })

        # Handle tools if provided
        if request.tools:
            logger.info(f"Tools provided in request: {len(request.tools)} tools")
            # Add tools to parameters if the model supports them
            params["tools"] = request.tools

            if request.tool_choice:
                params["tool_choice"] = request.tool_choice

        # Handle streaming
        if request.stream:
            async def generate_stream():
                # Start generation
                streamer = generate_text(model, tokenizer, prompt, params, stream=True, messages=formatted_messages)

                # Stream the results
                completion_id = f"chatcmpl-{uuid.uuid4()}"
                created = int(time.time())

                # Send the first chunk
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": config.model_name_or_path,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                # Stream the generated text
                collected_text = ""
                for text in streamer:
                    collected_text += text
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": config.model_name_or_path,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Send the final chunk
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": config.model_name_or_path,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(tokenizer.encode(prompt)) if tokenizer else count_tokens_gguf(model, prompt),
                        "completion_tokens": len(tokenizer.encode(collected_text)) if tokenizer else count_tokens_gguf(model, collected_text),
                        "total_tokens": (len(tokenizer.encode(prompt)) + len(tokenizer.encode(collected_text))) if tokenizer else (count_tokens_gguf(model, prompt) + count_tokens_gguf(model, collected_text)),
                    },
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Non-streaming request
            output = generate_text(model, tokenizer, prompt, params, stream=False, messages=formatted_messages)

            # Calculate token usage
            if tokenizer:
                prompt_tokens = len(tokenizer.encode(prompt))
                completion_tokens = len(tokenizer.encode(output))
            else:
                # Use GGUF token counting for llama-cpp-python models
                prompt_tokens = count_tokens_gguf(model, prompt)
                completion_tokens = count_tokens_gguf(model, output)

            # Get the actual model name used (could be client-requested model)
            model_name = request.model or config.model_name_or_path

            # Generate a unique ID for this completion
            completion_id = f"chatcmpl-{uuid.uuid4()}"

            # Create choices array based on number of completions requested
            choices = []

            # If multiple completions were requested (n > 1)
            if isinstance(output, list):
                for i, out in enumerate(output):
                    choices.append({
                        "index": i,
                        "message": {
                            "role": "assistant",
                            "content": out,
                        },
                        "finish_reason": "stop",
                    })
            else:
                # Single completion (default case)
                choices.append({
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output,
                    },
                    "finish_reason": "stop",
                })

            # Format response with all OpenAI-compatible fields
            return ChatCompletionResponse(
                id=completion_id,
                object="chat.completion",
                created=int(time.time()),
                model=model_name,
                choices=choices,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                # Add system fingerprint (OpenAI compatibility)
                system_fingerprint=f"fp_{model_name.replace('/', '_')}_{int(time.time())}",
            )

    # Add shutdown endpoint for graceful shutdown
    @app.post("/admin/shutdown")
    async def shutdown():
        # This requires additional security - you might want to restrict this further
        app.state.task_queue.shutdown()
        return {"status": "shutting down"}

    return app

def main():
    parser = argparse.ArgumentParser(description="HelpingAI Inference Server")

    # Add an auto-detection option
    parser.add_argument("--auto-configure", action="store_true",
                      help="Automatically detect and configure optimal settings for your hardware")

    # Model parameters
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", type=str, required=True,
                      help="Path to HuggingFace model or local model directory")
    model_group.add_argument("--model-revision", type=str, default=None,
                      help="Specific model revision to load")
    model_group.add_argument("--tokenizer", type=str, default=None,
                      help="Path to tokenizer (defaults to model path)")
    model_group.add_argument("--tokenizer-revision", type=str, default=None,
                      help="Specific tokenizer revision to load")

    # Server parameters
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to bind the server to")
    server_group.add_argument("--port", type=int, default=8000,
                      help="Port to bind the server to")

    # Hardware parameters
    hardware_group = parser.add_argument_group("Hardware Configuration")
    hardware_group.add_argument("--device", type=str, default="auto",
                      choices=["cuda", "cpu", "auto", "mps"],
                      help="Device to load the model on (cuda, cpu, auto, mps)")
    hardware_group.add_argument("--device-map", type=str, default="auto",
                      help="Device map for model distribution")
    hardware_group.add_argument("--dtype", type=str, default="float16",
                      choices=["float16", "float32", "bfloat16"],
                      help="Data type for model weights (float16 recommended for most hardware)")
    hardware_group.add_argument("--load-8bit", action="store_true",
                      help="Load model in 8-bit precision")
    hardware_group.add_argument("--load-4bit", action="store_true",
                      help="Load model in 4-bit precision")

    # TPU support
    tpu_group = parser.add_argument_group("TPU Configuration")
    tpu_group.add_argument("--use-tpu", action="store_true",
                      help="Enable TPU support (requires torch_xla)")
    tpu_group.add_argument("--tpu-cores", type=int, default=8,
                      help="Number of TPU cores to use")
    tpu_group.add_argument("--tpu-layers", type=int, default=-1,
                      help="Number of layers to offload to TPU (-1 means all)")
    tpu_group.add_argument("--tpu-bf16", action="store_true", default=True,
                      help="Use bfloat16 precision for TPU (recommended)")
    tpu_group.add_argument("--tpu-memory-limit", type=str, default="1GB",
                      help="Memory limit for TPU tensor allocator (e.g., '1GB', '2GB')")

    # API parameters
    api_group = parser.add_argument_group("API Configuration")
    api_group.add_argument("--api-keys", type=str, default="",
                      help="Comma-separated list of valid API keys")

    # Performance parameters
    perf_group = parser.add_argument_group("Performance Configuration")
    perf_group.add_argument("--max-concurrent", type=int, default=10,
                      help="Maximum number of concurrent requests")
    perf_group.add_argument("--max-queue", type=int, default=100,
                      help="Maximum queue size for pending requests")
    perf_group.add_argument("--timeout", type=int, default=60,
                      help="Timeout for requests in seconds")

    # GGUF support
    gguf_group = parser.add_argument_group("GGUF Model Configuration")
    gguf_group.add_argument("--enable-gguf", action="store_true",
                      help="Enable GGUF model support (requires llama-cpp-python)")
    gguf_group.add_argument("--gguf-path", type=str, default=None,
                      help="Path to GGUF model file (can be used with TPU if needed)")
    gguf_group.add_argument("--download-gguf", action="store_true",
                      help="Download GGUF model from Hugging Face (if available)")
    gguf_group.add_argument("--gguf-filename", type=str, default=None,
                      help="Specific GGUF filename to download (e.g., 'model-q4_k_m.gguf')")
    gguf_group.add_argument("--num-gpu-layers", type=int, default=-1,
                      help="Number of GPU layers for GGUF models (-1 means all)")

    args = parser.parse_args()

    # Apply auto-configuration if requested
    if args.auto_configure:
        logger.info("Auto-configure flag detected, performing hardware detection...")

        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            logger.info("NVIDIA GPU detected, configuring for optimal performance")
            args.device = "cuda"

            # Get GPU memory to determine quantization
            try:
                props = torch.cuda.get_device_properties(0)
                gpu_memory_gb = props.total_memory / (1024**3)
                logger.info(f"GPU: {props.name} with {gpu_memory_gb:.2f}GB VRAM")

                # Auto-configure based on available VRAM
                if gpu_memory_gb < 8:
                    logger.info("Low VRAM GPU detected, enabling 4-bit quantization")
                    args.load_4bit = True
                elif gpu_memory_gb < 16:
                    logger.info("Medium VRAM GPU detected, enabling 8-bit quantization")
                    args.load_8bit = True
                else:
                    logger.info(f"High VRAM GPU detected ({gpu_memory_gb:.2f}GB), using FP16")

                # Set compute capability-specific optimizations
                compute_capability = f"{props.major}.{props.minor}"
                logger.info(f"CUDA Compute Capability: {compute_capability}")

                # For multi-GPU systems, use device map
                if torch.cuda.device_count() > 1:
                    logger.info(f"Multiple GPUs detected ({torch.cuda.device_count()}), using auto device map")
                    args.device_map = "auto"
            except Exception as e:
                logger.warning(f"Error during GPU detection: {e}")

        # Check for Apple Silicon (MPS)
        elif hasattr(torch, 'has_mps') and torch.backends.mps.is_available():
            logger.info("Apple Silicon GPU detected, configuring for MPS")
            args.device = "mps"

            # For Apple Silicon, GGUF with Metal is often best
            logger.info("Recommending GGUF with Metal for Apple Silicon")
            if not args.enable_gguf and "gguf" not in args.model.lower():
                logger.info("Auto-enabling GGUF for better performance on Apple Silicon")
                args.enable_gguf = True

                # If no specific GGUF file is specified but we're downloading, suggest a good default
                if args.download_gguf and not args.gguf_filename:
                    logger.info("Auto-selecting a 4-bit quantization GGUF file for best performance")
                    # Look for a 4-bit quantized version as a good default
                    args.gguf_filename = "model-q4_k_m.gguf"

        # Check for TPU
        elif os.environ.get("TPU_NAME") or os.environ.get("COLAB_TPU_ADDR"):
            logger.info("Google Cloud TPU environment detected")
            if XLA_AVAILABLE:
                logger.info("TPU support confirmed with torch_xla, enabling TPU mode")
                args.use_tpu = True
                args.device = "xla"
            else:
                logger.warning("TPU environment detected but torch_xla not installed")
                logger.warning("Install with: pip install torch_xla")
                args.device = "cpu"

        # Fallback to CPU
        else:
            logger.info("No GPU/TPU detected, configuring for CPU")
            args.device = "cpu"

            # For CPU inference, GGUF is strongly recommended
            logger.info("For CPU inference, GGUF models are recommended for performance")
            if not args.enable_gguf and "gguf" not in args.model.lower():
                logger.info("Auto-enabling GGUF for CPU inference")
                args.enable_gguf = True
                args.download_gguf = True

                # If no specific GGUF file is specified but we're downloading, suggest a good default
                if not args.gguf_filename:
                    logger.info("Auto-selecting a 4-bit quantization GGUF file for CPU")
                    args.gguf_filename = "model-q4_k_m.gguf"

    # Enhanced model path handling - check if the model is a local path or HF model ID
    if not args.model.startswith(("https://", "http://")) and "/" in args.model and not os.path.exists(args.model):
        logger.info(f"Model path looks like a Hugging Face model ID: {args.model}")

        # If GGUF is enabled but no filename specified, try to auto-detect common GGUF filenames
        if args.enable_gguf and args.download_gguf and not args.gguf_filename:
            logger.info("Auto-detecting GGUF filename for the model")
            common_gguf_files = [
                "model-q4_k_m.gguf",  # Most common balanced quantization
                "model-q5_k_m.gguf",  # Higher quality
                "model-Q4_K_M.gguf",  # Alternative naming
                "model-Q5_K_M.gguf",
                "model.gguf",         # Generic name
                "ggml-model-q4_k.gguf",
                "ggml-model-q5_k.gguf",
            ]

            # For Apple Silicon, suggest Metal-optimized GGUF files if available
            if args.device == "mps":
                common_gguf_files = ["model-q4_k_m.gguf", "model-q5_k_m.gguf"] + common_gguf_files

            # For CPU, prioritize smallest files
            if args.device == "cpu":
                common_gguf_files = ["model-q4_0.gguf", "model-q4_k_m.gguf"] + common_gguf_files

            # Try to check which GGUF files exist for this model
            try:
                api_url = f"https://huggingface.co/api/models/{args.model}"
                response = requests.get(api_url)
                if response.status_code == 200:
                    model_info = response.json()
                    siblings = model_info.get("siblings", [])
                    available_files = [s["rfilename"] for s in siblings]

                    # Filter for GGUF files
                    gguf_files = [f for f in available_files if f.endswith('.gguf')]

                    if gguf_files:
                        # Log available GGUF files
                        logger.info(f"Found {len(gguf_files)} GGUF files for this model")
                        for i, file in enumerate(gguf_files[:5]):  # Show first 5
                            logger.info(f"  - {file}")
                        if len(gguf_files) > 5:
                            logger.info(f"  ... and {len(gguf_files) - 5} more")

                        # Try to find a good match from common files
                        for common_file in common_gguf_files:
                            if common_file in gguf_files:
                                args.gguf_filename = common_file
                                logger.info(f"Auto-selected GGUF file: {common_file}")
                                break

                        # If no match found, use the first available GGUF file
                        if not args.gguf_filename:
                            args.gguf_filename = gguf_files[0]
                            logger.info(f"Using first available GGUF file: {args.gguf_filename}")
            except Exception as e:
                logger.warning(f"Error checking for GGUF files: {e}")
                logger.warning("You may need to specify --gguf-filename manually")

    # Create server config
    config = ServerConfig(
        model_name_or_path=args.model,
        model_revision=args.model_revision,
        tokenizer_name_or_path=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        host=args.host,
        port=args.port,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        api_keys=args.api_keys.split(",") if args.api_keys else [],
        max_concurrent_requests=args.max_concurrent,
        max_queue_size=args.max_queue,
        timeout=args.timeout,
        enable_gguf=args.enable_gguf,
        gguf_path=args.gguf_path,
        gguf_filename=args.gguf_filename,
        download_gguf=args.download_gguf,
        num_gpu_layers=args.num_gpu_layers,
        use_tpu=args.use_tpu,
        tpu_cores=args.tpu_cores,
        tpu_layers=args.tpu_layers,
        tpu_bf16=args.tpu_bf16,
        tpu_memory_limit=args.tpu_memory_limit,
    )

    # Create the FastAPI app
    app = create_app(config)

    # Add routes
    app = create_routes(app)

    # Preload the model
    model, tokenizer = load_model(config)

    # Start the server
    logger.info(f"Starting HelpingAI Inference Server v{SERVER_VERSION}")
    logger.info(f"Model: {config.model_name_or_path}")

    # Log hardware configuration
    if config.use_tpu:
        logger.info(f"Hardware: TPU with {config.tpu_cores} cores")
    else:
        logger.info(f"Device: {config.device}")

    # Log model format
    if config.enable_gguf:
        if config.download_gguf:
            logger.info(f"Model format: GGUF (auto-downloaded from Hugging Face)")
            if config.gguf_filename:
                logger.info(f"Requested GGUF file: {config.gguf_filename}")
        elif config.gguf_path:
            logger.info(f"Model format: GGUF from {config.gguf_path}")
        else:
            logger.info("Model format: GGUF (will attempt to download if needed)")
    else:
        logger.info(f"Model format: fp16 (transformers)")
        if config.load_8bit:
            logger.info("Quantization: 8-bit")
        elif config.load_4bit:
            logger.info("Quantization: 4-bit")

    logger.info(f"Chat template: {'Using model-specific template when available' if hasattr(tokenizer, 'chat_template') else 'Using default template'}")
    logger.info(f"API Authentication: {'Enabled' if config.api_keys else 'Disabled'}")

    uvicorn.run(app, host=config.host, port=config.port)

if __name__ == "__main__":
    main()
"""
pip install -U torch transformers fastapi uvicorn pydantic

pip install -U llama-cpp-python (optional, for GGUF support)

python helpingai_server.py --model HelpingAI/HelpingAI-15B

Additional Configuration Options

To use a quantized version for better performance:

python helpingai_server.py --model HelpingAI/HelpingAI-15B --load-8bit

To use GGUF models (like the quantized versions mentioned in your model card):

python helpingai_server.py --enable-gguf --gguf-path path/to/helpingai-15b-q4_k_m.gguf

To add API key security:

python helpingai_server.py --model HelpingAI/HelpingAI-15B --api-keys "key1,key2,key3"

Making API Requests

The server implements OpenAI-compatible endpoints for easy integration:

Completions API:

curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "I feel really happy today because",
    "max_tokens": 100,
    "temperature": 0.7
  }'
Chat Completions API:


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
"""