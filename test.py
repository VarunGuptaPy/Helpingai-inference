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
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import threading
import queue

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

# Default template for HelpingAI models
DEFAULT_TEMPLATE = """
<|im_start|>system: {system}
<|im_end|>
<|im_start|>user: {user}
<|im_end|>
<|im_start|>assistant:
"""

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
    messages: List[ChatMessage]
    system_prompt: Optional[str] = "You are HelpingAI, an emotionally intelligent AI assistant."
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    user_id: Optional[str] = None

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

class ModelsResponse(BaseModel):
    object: str = Field("list", description="Object type")
    data: List[Dict[str, Any]] = Field(..., description="List of available models")

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
    
    def __post_init__(self):
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path
            
        if not torch.cuda.is_available() and self.device == "cuda":
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
    
    if config.enable_gguf and config.gguf_path:
        # GGUF loading logic requires llama-cpp-python
        try:
            from llama_cpp import Llama
            
            model = Llama(
                model_path=config.gguf_path,
                n_gpu_layers=config.num_gpu_layers,
                n_ctx=4096,  # Context size
                verbose=False
            )
            tokenizer = None  # GGUF models use their own tokenization
            
            logger.info(f"Loaded GGUF model from {config.gguf_path}")
            
        except ImportError:
            logger.error("llama-cpp-python not installed. Cannot load GGUF model.")
            raise
    else:
        # Load with transformers
        model_kwargs = {
            "device_map": config.device_map,
            "torch_dtype": torch.float16 if config.dtype == "float16" else torch.float32,
            "use_cache": config.use_cache,
        }
        
        if config.load_8bit:
            model_kwargs["load_in_8bit"] = True
        
        if config.load_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            model_kwargs["bnb_4bit_quant_type"] = "nf4"
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            revision=config.model_revision,
            **model_kwargs
        )
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path, 
            revision=config.tokenizer_revision,
            use_fast=True
        )
        
        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Model loaded successfully on {config.device}")
    
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
    """Format messages into the appropriate prompt format."""
    if not messages:
        return system_prompt
    
    # Extract user message from the last user message
    user_message = None
    for msg in reversed(messages):
        if msg.role.lower() == "user":
            user_message = msg.content
            break
    
    if user_message is None:
        return system_prompt
    
    # Format according to the HelpingAI template
    prompt = DEFAULT_TEMPLATE.format(
        system=system_prompt,
        user=user_message
    )
    
    return prompt

def generate_text(model, tokenizer, prompt, params, stream=False):
    """Generate text using the model."""
    
    # Check if using GGUF model
    if isinstance(model, object) and hasattr(model, "generate") and tokenizer is None:
        # Using llama-cpp-python for GGUF models
        output = model.generate(
            prompt, 
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"],
            stop=params["stop"] if params["stop"] else [],
            stream=stream
        )
        
        if stream:
            return output  # This is a generator for streaming
        else:
            return output["choices"][0]["text"]
    
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
        
        # Handle streaming
        if request.stream:
            async def generate_stream():
                # Start generation
                streamer = generate_text(model, tokenizer, prompt, params, stream=True)
                
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
                        "prompt_tokens": len(tokenizer.encode(prompt)) if tokenizer else 0,
                        "completion_tokens": len(tokenizer.encode(collected_text)) if tokenizer else 0,
                        "total_tokens": (len(tokenizer.encode(prompt)) + len(tokenizer.encode(collected_text))) if tokenizer else 0,
                    },
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Non-streaming request
            output = generate_text(model, tokenizer, prompt, params, stream=False)
            
            # Calculate token usage
            prompt_tokens = len(tokenizer.encode(prompt)) if tokenizer else 0
            completion_tokens = len(tokenizer.encode(output)) if tokenizer else 0
            
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
        model, tokenizer = load_model(config)
        
        # Define parameters
        params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop": request.stop,
        }
        
        # Format messages into prompt
        prompt = format_prompt(request.messages, request.system_prompt, tokenizer)
        
        # Handle streaming
        if request.stream:
            async def generate_stream():
                # Start generation
                streamer = generate_text(model, tokenizer, prompt, params, stream=True)
                
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
                        "prompt_tokens": len(tokenizer.encode(prompt)) if tokenizer else 0,
                        "completion_tokens": len(tokenizer.encode(collected_text)) if tokenizer else 0,
                        "total_tokens": (len(tokenizer.encode(prompt)) + len(tokenizer.encode(collected_text))) if tokenizer else 0,
                    },
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Non-streaming request
            output = generate_text(model, tokenizer, prompt, params, stream=False)
            
            # Calculate token usage
            prompt_tokens = len(tokenizer.encode(prompt)) if tokenizer else 0
            completion_tokens = len(tokenizer.encode(output)) if tokenizer else 0
            
            # Format response
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                object="chat.completion",
                created=int(time.time()),
                model=config.model_name_or_path,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output,
                        },
                        "finish_reason": "stop",
                    }
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
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
                      help="Device to load the model on (cuda, cpu)")
    parser.add_argument("--device-map", type=str, default="auto",
                      help="Device map for model distribution")
    parser.add_argument("--dtype", type=str, default="float16",
                      choices=["float16", "float32"],
                      help="Data type for model weights")
    parser.add_argument("--load-8bit", action="store_true",
                      help="Load model in 8-bit precision")
    parser.add_argument("--load-4bit", action="store_true",
                      help="Load model in 4-bit precision")
    
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
                      help="Path to GGUF model file")
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
        num_gpu_layers=args.num_gpu_layers,
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
    logger.info(f"Device: {config.device}")
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