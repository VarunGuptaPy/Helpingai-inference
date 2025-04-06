#!/usr/bin/env python3
"""
Test script for the TPU backend
"""

import os
import sys
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("tpu-backend-test")

def test_tpu_backend(model_path, prompt, max_tokens=100):
    """Test the TPU backend with a simple prompt"""
    try:
        # Import PyTorch/XLA
        import torch_xla.core.xla_model as xm # type: ignore
        devices = xm.get_xla_supported_devices()
        logger.info(f"Found {len(devices)} TPU devices")

        # Set TPU-specific environment variables
        os.environ['XLA_USE_BF16'] = '1'  # Enable bfloat16 for better performance
        os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '1000000000'  # 1GB
        os.environ['GGML_TPU_ENABLE'] = '1'
        os.environ['GGML_TPU_LAYERS'] = '-1'  # Use all layers on TPU

        # Import llama-cpp-python
        from llama_cpp import Llama

        # Initialize the model with TPU support
        logger.info(f"Loading model from {model_path}")
        model = Llama(
            model_path=model_path,
            n_ctx=2048,
            use_mlock=True,
            n_gpu_layers=-1,
            use_tpu=True,
            verbose=True
        )

        # Generate text
        logger.info(f"Generating text for prompt: {prompt}")
        output = model(prompt, max_tokens=max_tokens)

        # Print the output
        logger.info("Generation complete!")
        logger.info(f"Output: {output['choices'][0]['text']}")

        return True

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure PyTorch/XLA and llama-cpp-python are installed")
        return False

    except Exception as e:
        logger.error(f"Error: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the TPU backend")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model file")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?",
                        help="Prompt to generate text from")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")

    args = parser.parse_args()

    # Test the TPU backend
    success = test_tpu_backend(args.model, args.prompt, args.max_tokens)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
