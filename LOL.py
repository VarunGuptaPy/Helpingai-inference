import requests
import json

def make_chat_request(model=None, json_mode=False, stream=False):
    # API endpoint
    url = "http://localhost:8000/v1/chat/completions"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "your_api_key"
    }
    
    # System prompt
    system_prompt = "You are an advanced AI assistant."
    
    # Request body with OpenAI-compatible parameters
    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "How many r in strawberry?"
            }
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "n": 1,  # Number of completions to generate
        "stream": stream,  # Enable streaming if requested
        "user": "example_user"  # User identifier
    }
    
    # Add model if specified
    if model:
        payload["model"] = model
        
    # Add JSON mode if requested
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        # Make the POST request
        with requests.post(url, headers=headers, json=payload, stream=stream) as response:
            # Check if request was successful
            response.raise_for_status()
            
            if stream:
                # Stream the response line by line
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        data = json.loads(line)
                        print(json.dumps(data, indent=2))  # Print each chunk of streamed data
            else:
                # Return the JSON response for non-streaming
                return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

if __name__ == "__main__":
    
    # Streaming request
    print("\nStreaming Request:")
    make_chat_request(stream=True)