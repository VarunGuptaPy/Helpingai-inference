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

        # Check for available hardware and set appropriate device
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
                self.device = "mps"
            else:
                logger.warning("MPS (Apple Silicon GPU) requested but not available, falling back to CPU")
                self.device = "cpu"
        # Check for CUDA
        elif not self.use_tpu and self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

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

    # First load the tokenizer as it's needed for both GGUF and transformers paths
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            revision=config.tokenizer_revision,
            use_fast=True
        )

        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Tokenizer loaded successfully from {config.tokenizer_name_or_path}")
    except Exception as e:
        logger.warning(f"Error loading tokenizer: {e}")
        tokenizer = None

    # Handle GGUF model path - download from HF if requested
    gguf_path = config.gguf_path
    if config.download_gguf:
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

            # Configure GGUF model based on available hardware
            gguf_kwargs = {
                "model_path": gguf_path,
                "n_ctx": 4096,  # Context size
                "verbose": False
            }

            # Configure GPU usage for GGUF
            if torch.cuda.is_available() and config.device == "cuda":
                gguf_kwargs["n_gpu_layers"] = config.num_gpu_layers
                gguf_kwargs["use_mlock"] = True  # Helps with GPU memory management
                logger.info(f"Configuring GGUF model for GPU with {config.num_gpu_layers} layers")
            elif config.use_tpu:
                # TPU-specific configuration for GGUF
                logger.info("Configuring GGUF model for TPU")
                gguf_kwargs["n_gpu_layers"] = config.num_gpu_layers if config.num_gpu_layers > 0 else -1
                gguf_kwargs["use_mlock"] = True
                gguf_kwargs["use_tpu"] = True

                # Set environment variables for TPU support
                os.environ["GGML_TPU_ENABLE"] = "1"
                os.environ["GGML_TPU_LAYERS"] = str(config.num_gpu_layers if config.num_gpu_layers > 0 else -1)

                logger.info(f"TPU configuration: using {config.num_gpu_layers if config.num_gpu_layers > 0 else 'all'} layers on TPU")
            else:
                # CPU configuration
                gguf_kwargs["n_threads"] = os.cpu_count()
                logger.info(f"Configuring GGUF model for CPU with {os.cpu_count()} threads")

            model = Llama(**gguf_kwargs)
            logger.info(f"Loaded GGUF model from {gguf_path}")

        except ImportError:
            logger.error("llama-cpp-python not installed. Cannot load GGUF model.")
            if config.use_tpu or gguf_path:
                logger.error("Attempting to fall back to transformers with fp16...")
                config.enable_gguf = False
            else:
                raise

    # Transformers path
    if not config.enable_gguf or not gguf_path:
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
            model_kwargs["load_in_8bit"] = True

        if config.load_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            model_kwargs["bnb_4bit_quant_type"] = "nf4"

        # Load the model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                revision=config.model_revision,
                **model_kwargs
            )

            # Move model to MPS if using MPS
            if config.device == "mps" and hasattr(torch, 'has_mps') and torch.has_mps and torch.backends.mps.is_available():
                model = model.to(torch.device("mps"))
                logger.info("Model moved to MPS device")
            # Move model to TPU if using TPU
            elif config.use_tpu and 'device' in locals():
                model = model.to(device)

            logger.info(f"Model loaded successfully on {config.device}")
        except Exception as e:
            logger.error(f"Error loading model with transformers: {e}")
            raise

    # Cache the model and tokenizer
    MODEL_CACHE[model_id] = model
    TOKENIZER_CACHE[model_id] = tokenizer

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

    # Model parameters
    parser.add_argument("--model", type=str, required=True,
                      help="Path to HuggingFace model or local model directory")
    parser.add_argument("--model-revision", type=str, default=None,
                      help="Specific model revision to load")
    parser.add_argument("--tokenizer", type=str, default=None,
                      help="Path to tokenizer (defaults to model path)")
    parser.add_argument("--tokenizer-revision", type=str, default=None,
                      help="Specific tokenizer revision to load")

    # Server parameters
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000,
                      help="Port to bind the server to")

    # Hardware parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      choices=["cuda", "cpu", "auto", "mps"],
                      help="Device to load the model on (cuda, cpu, auto, mps)")
    parser.add_argument("--device-map", type=str, default="auto",
                      help="Device map for model distribution")
    parser.add_argument("--dtype", type=str, default="float16",
                      choices=["float16", "float32", "bfloat16"],
                      help="Data type for model weights (float16 recommended for most hardware)")
    parser.add_argument("--load-8bit", action="store_true",
                      help="Load model in 8-bit precision")
    parser.add_argument("--load-4bit", action="store_true",
                      help="Load model in 4-bit precision")

    # TPU support
    parser.add_argument("--use-tpu", action="store_true",
                      help="Enable TPU support (requires torch_xla)")
    parser.add_argument("--tpu-cores", type=int, default=8,
                      help="Number of TPU cores to use")
    parser.add_argument("--tpu-layers", type=int, default=-1,
                      help="Number of layers to offload to TPU (-1 means all)")
    parser.add_argument("--tpu-bf16", action="store_true", default=True,
                      help="Use bfloat16 precision for TPU (recommended)")
    parser.add_argument("--tpu-memory-limit", type=str, default="1GB",
                      help="Memory limit for TPU tensor allocator (e.g., '1GB', '2GB')")

    # API parameters
    parser.add_argument("--api-keys", type=str, default="",
                      help="Comma-separated list of valid API keys")

    # Performance parameters
    parser.add_argument("--max-concurrent", type=int, default=10,
                      help="Maximum number of concurrent requests")
    parser.add_argument("--max-queue", type=int, default=100,
                      help="Maximum queue size for pending requests")
    parser.add_argument("--timeout", type=int, default=60,
                      help="Timeout for requests in seconds")

    # GGUF support
    parser.add_argument("--enable-gguf", action="store_true",
                      help="Enable GGUF model support (requires llama-cpp-python)")
    parser.add_argument("--gguf-path", type=str, default=None,
                      help="Path to GGUF model file (can be used with TPU if needed)")
    parser.add_argument("--download-gguf", action="store_true",
                      help="Download GGUF model from Hugging Face (if available)")
    parser.add_argument("--gguf-filename", type=str, default=None,
                      help="Specific GGUF filename to download (e.g., 'model-q4_k_m.gguf')")
    parser.add_argument("--num-gpu-layers", type=int, default=-1,
                      help="Number of GPU layers for GGUF models (-1 means all)")

    args = parser.parse_args()

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