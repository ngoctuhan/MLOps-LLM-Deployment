# Llama.cpp GGUF Server

This repository contains a Docker Compose configuration for running a Llama.cpp server with GGUF model support.

## Features

- GPU-accelerated inference using CUDA
- Simple API for text generation
- Easy deployment with Docker Compose
- Configurable model parameters

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with appropriate drivers
- NVIDIA Container Toolkit installed

## Setup Instructions

### 1. Install NVIDIA Container Toolkit

If you haven't already installed the NVIDIA Container Toolkit, follow these steps:

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### 2. Directory Structure

Create the following directory structure:

```
project-root/
├── docker-compose.yml
├── README.md
└── models/
    └── Llama-3.2-3B-Instruct-Q6_K.gguf
```

### 3. Download the GGUF Model

Download your preferred GGUF model and place it in the `llm_serving/llama_cpp/models/` directory. For example:

```bash
mkdir -p models/
cd models/
# Download your model here. For example:
wget https://huggingface.co/TheBloke/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf
```

## Docker Compose Configuration

## Usage

### Starting the Server

To start the Llama.cpp server:

```bash
docker-compose up -d
```

This will run the server in detached mode. The server will be accessible at `http://localhost:8880`.

### API Endpoints

The server provides the following OpenAI-compatible API endpoints:

#### Chat Completion API

```
POST /v1/chat/completions
```

Example curl request:

```bash
curl --location 'http://localhost:8880/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer no-key' \
--data '{
  "model": "gpt-xyz",
  "messages": [
    {
      "role": "system",
      "content": "Kiểm tra chính tả và nghĩa của văn bản tiếng Việt. \n\nRule:\n- By by number \n- By pass special character\n\nOnly response JSON format and nothing:\n\n{ '\''status'\'': True | False\n  '\''explain'\'': \"<explain>\"\n}\n\nExplain:\n\nQ: \"Hôm nai trời đẹp\"\nResponse: \n{\n  \"status\": False, \n  \"text\": \"nai -> nay\"\n}\n\nQ: \"TIẾNG VIỆT\"\nResponse: \n{\n  \"status\": True, \n  \"text\": \"nai -> nay\"\n}"
    },
    {
      "role": "user",
      "content": "Text: '\''Tìm hình ứng với mỗi tiếng'\''"
    }
  ]
}'
```

#### Using with OpenAI Libraries

You can use the server with popular OpenAI client libraries by setting the base URL to your server address. Examples for different programming languages:

##### Python

```python
from openai import OpenAI

client = OpenAI(
    api_key="no-key",  # Can be any string as the server doesn't verify API keys
    base_url="http://localhost:8880"
)

response = client.chat.completions.create(
    model="gpt-xyz",  # Model name can be any string
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```

##### JavaScript/Node.js

```javascript
const { OpenAI } = require('openai');

const openai = new OpenAI({
  apiKey: 'no-key',
  baseURL: 'https://llm.monkey.edu.vn'
});

async function main() {
  const response = await openai.chat.completions.create({
    model: 'gpt-xyz',
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Hello, how are you?' }
    ]
  });

  console.log(response.choices[0].message.content);
}

main();
```

### Monitoring Logs

To view server logs:

```bash
docker-compose logs -f llama-server
```

### Stopping the Server

To stop the server:

```bash
docker-compose down
```

## Configuration Parameters

Here's what each parameter in the command does:

- `-m /models/Llama-3.2-3B-Instruct-Q6_K.gguf`: Specifies the model file to load
- `-c 2048`: Sets the context size (token window) to 2048 tokens
- `--host 0.0.0.0`: Binds the server to all network interfaces
- `--port 8080`: Sets the internal port (mapped to 8880 externally)
- `--n-gpu-layers 64`: Offloads 64 layers to the GPU for acceleration

## Troubleshooting

### CUDA/GPU Issues

If you encounter GPU-related issues, verify that:

1. NVIDIA drivers are properly installed
2. NVIDIA Container Toolkit is installed and configured
3. Your GPU has enough VRAM for the model you're using

Check GPU status with:

```bash
nvidia-smi
```

### Memory Issues

If the server crashes due to memory issues:

1. Try a smaller model (e.g., a more quantized version)
2. Reduce the context size (`-c` parameter)
3. Reduce the number of GPU layers (`--n-gpu-layers` parameter)

## Additional Resources

- [Llama.cpp GitHub Repository](https://github.com/ggerganov/llama.cpp)
- [GGUF Models on Hugging Face](https://huggingface.co/models?search=gguf)
- [Llama.cpp Server API Documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)