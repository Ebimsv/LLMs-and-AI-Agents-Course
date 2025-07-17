# Day 2: LLM APIs with Python - OpenAI, Ollama, and Beyond

## 🎯 Learning Objectives

By the end of this day, you will be able to:
- Understand REST APIs and how LLM providers expose their services
- Integrate with OpenAI's GPT models via API
- Set up and use Ollama for local LLM deployment
- Handle API errors and implement best practices
- Optimize costs and manage API usage efficiently
- Compare different LLM providers and their capabilities

## 🔌 API Fundamentals

### What is an API?
An Application Programming Interface (API) is a set of rules that allows different software applications to communicate with each other. For LLMs, APIs provide a standardized way to send text inputs and receive generated responses.

### REST API Basics
- **HTTP Methods**: GET, POST, PUT, DELETE
- **Endpoints**: URLs that accept requests
- **Authentication**: API keys, tokens, or OAuth
- **Request/Response**: JSON format for data exchange

## 🚀 OpenAI Integration

### Setup and Configuration

1. **Get an API Key**
   - Sign up at [OpenAI Platform](https://platform.openai.com/)
   - Navigate to API Keys section
   - Create a new secret key

2. **Install the SDK**
   ```bash
   pip install openai python-dotenv
   ```

3. **Environment Setup**
   ```python
   import os
   from openai import OpenAI
   from dotenv import load_dotenv

   load_dotenv()
   client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
   ```

### Basic Usage Examples

#### Simple Chat Completion
```python
def simple_chat(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content

# Usage
result = simple_chat("Explain quantum computing in simple terms")
print(result)
```

#### Multi-turn Conversation
```python
def chat_conversation():
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    
    # Add assistant's response to conversation
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    
    # Continue conversation
    messages.append({"role": "user", "content": "Now optimize it for performance"})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    return response.choices[0].message.content
```

#### Streaming Responses
```python
def stream_response(prompt):
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
```

## 🏠 Ollama Local Setup

### Installation

#### macOS/Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows
Download from [Ollama.ai](https://ollama.ai/download)

### Basic Usage

1. **Start Ollama Service**
   ```bash
   ollama serve
   ```

2. **Pull a Model**
   ```bash
   ollama pull llama2
   ollama pull codellama
   ollama pull mistral
   ```

3. **Python Integration**
   ```python
   import requests
   import json

   def ollama_chat(prompt, model="llama2"):
       url = "http://localhost:11434/api/generate"
       data = {
           "model": model,
           "prompt": prompt,
           "stream": False
       }
       
       response = requests.post(url, json=data)
       return response.json()["response"]

   # Usage
   result = ollama_chat("Explain machine learning")
   print(result)
   ```

### Advanced Ollama Features

#### Model Management
```python
def list_models():
    response = requests.get("http://localhost:11434/api/tags")
    return response.json()["models"]

def model_info(model_name):
    response = requests.get(f"http://localhost:11434/api/show", 
                          json={"name": model_name})
    return response.json()
```

#### Custom Model Configuration
```python
def create_custom_model():
    # Create a Modelfile
    modelfile_content = """
FROM llama2
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM You are a helpful coding assistant.
"""
    
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)
    
    # Create the model
    import subprocess
    subprocess.run(["ollama", "create", "my-coder", "-f", "Modelfile"])
```

## 🔧 Error Handling and Best Practices

### Robust API Client
```python
import time
import logging
from typing import Optional, Dict, Any

class LLMClient:
    def __init__(self, provider: str, api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
    def chat_completion(self, 
                       messages: list, 
                       model: str = "gpt-3.5-turbo",
                       max_retries: int = 3,
                       **kwargs) -> Optional[str]:
        
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    return self._openai_completion(messages, model, **kwargs)
                elif self.provider == "ollama":
                    return self._ollama_completion(messages, model, **kwargs)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All attempts failed: {str(e)}")
                    return None
    
    def _openai_completion(self, messages: list, model: str, **kwargs) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    
    def _ollama_completion(self, messages: list, model: str, **kwargs) -> str:
        # Convert chat format to Ollama format
        prompt = self._format_messages_for_ollama(messages)
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=data)
        response.raise_for_status()
        return response.json()["response"]
    
    def _format_messages_for_ollama(self, messages: list) -> str:
        formatted = ""
        for msg in messages:
            if msg["role"] == "system":
                formatted += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                formatted += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted += f"Assistant: {msg['content']}\n"
        return formatted
```

### Usage Example
```python
# Initialize clients
openai_client = LLMClient("openai", api_key=os.getenv("OPENAI_API_KEY"))
ollama_client = LLMClient("ollama")

# Test both providers
messages = [{"role": "user", "content": "Write a hello world program in Python"}]

openai_result = openai_client.chat_completion(messages)
ollama_result = ollama_client.chat_completion(messages, model="codellama")

print("OpenAI:", openai_result)
print("Ollama:", ollama_result)
```

## 💰 Cost Optimization

### Token Management
```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_cost(tokens: int, model: str = "gpt-3.5-turbo") -> float:
    # Approximate costs per 1K tokens (as of 2024)
    costs = {
        "gpt-4": 0.03,      # $0.03 per 1K input tokens
        "gpt-3.5-turbo": 0.002,  # $0.002 per 1K input tokens
    }
    return (tokens / 1000) * costs.get(model, 0.002)

# Usage
text = "Your input text here"
token_count = count_tokens(text)
cost = estimate_cost(token_count)
print(f"Tokens: {token_count}, Estimated cost: ${cost:.4f}")
```

### Batch Processing
```python
def batch_process(prompts: list, batch_size: int = 5):
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        
        # Process batch
        batch_results = []
        for prompt in batch:
            result = openai_client.chat_completion([{"role": "user", "content": prompt}])
            batch_results.append(result)
        
        results.extend(batch_results)
        
        # Rate limiting
        time.sleep(1)
    
    return results
```

## 📊 Provider Comparison

| Feature | OpenAI | Ollama | Anthropic | Hugging Face |
|---------|--------|--------|-----------|--------------|
| **Cost** | Pay-per-token | Free (local) | Pay-per-token | Free/Paid |
| **Privacy** | Data sent to OpenAI | Local only | Data sent to Anthropic | Varies |
| **Speed** | Fast | Depends on hardware | Fast | Varies |
| **Models** | GPT-3.5, GPT-4 | Many open-source | Claude | Thousands |
| **Setup** | API key only | Local installation | API key only | Varies |

## 🎯 Exercises

### Exercise 1: Multi-Provider Testing
1. Set up both OpenAI and Ollama
2. Test the same prompt on both providers
3. Compare response quality, speed, and cost
4. Document your findings

### Exercise 2: Error Handling
1. Implement robust error handling for API calls
2. Test with invalid API keys, network issues
3. Implement retry logic with exponential backoff
4. Create a monitoring system for API usage

### Exercise 3: Cost Optimization
1. Implement token counting for your use cases
2. Create a cost estimation tool
3. Optimize prompts to reduce token usage
4. Set up usage alerts and limits

## 🔍 Advanced Topics

### Async Processing
```python
import asyncio
import aiohttp

async def async_chat_completion(session, messages, model="gpt-3.5-turbo"):
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
        json={
            "model": model,
            "messages": messages
        }
    ) as response:
        result = await response.json()
        return result["choices"][0]["message"]["content"]

async def process_multiple_requests(prompts):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            task = async_chat_completion(session, messages)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
```

### Custom Model Fine-tuning
```python
# Example of preparing data for fine-tuning
def prepare_fine_tuning_data(conversations):
    formatted_data = []
    
    for conversation in conversations:
        formatted_conversation = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                *conversation
            ]
        }
        formatted_data.append(formatted_conversation)
    
    return formatted_data
```

## 🚀 Next Steps

After completing Day 2, you'll be ready to:
- Move to Day 3: Deep dive into Ollama and local LLM deployment
- Build more complex applications with multiple providers
- Implement production-ready error handling
- Optimize your applications for cost and performance

## 📖 Additional Resources

### Documentation
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Ollama Documentation](https://ollama.ai/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)

### Tools and Libraries
- [tiktoken](https://github.com/openai/tiktoken) - Token counting
- [openai-python](https://github.com/openai/openai-python) - Official SDK
- [ollama-python](https://github.com/ollama/ollama-python) - Python client

### Best Practices
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [API Rate Limiting](https://platform.openai.com/docs/guides/rate-limits)
- [Error Handling Guide](https://platform.openai.com/docs/guides/error-codes)

---

**Ready for Day 3?** 🚀

In the next tutorial, we'll explore Ollama in depth and learn how to run LLMs locally with just one command! 