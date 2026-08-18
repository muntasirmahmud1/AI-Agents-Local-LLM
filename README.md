# Local LLM Setup for PANCAM on NVIDIA Jetson AGX Orin

This guide describes how to install and run a local GPU-accelerated LLM backend for the PANCAM software using:

- NVIDIA Jetson AGX Orin
- Jetson Linux R36.5
- CUDA 12.6
- `llama.cpp`
- `Qwen3.5-9B-Q4_K_M.gguf`
- `llama-server`

The local model runs independently from the PANCAM GUI. The GUI will later communicate with `llama-server` through a local OpenAI-compatible HTTP API.

## Architecture

```text
PANCAM GUI
    |
    | HTTP request
    v
http://127.0.0.1:8080
    |
    v
llama-server
    |
    v
Qwen3.5-9B-Q4_K_M
    |
    v
Jetson CUDA GPU
```

This design makes it possible to replace the local backend with a paid cloud API later without rewriting the agent or GUI logic.

---

## 1. Verify the Jetson environment

Run:

```bash
uname -m
cat /etc/nv_tegra_release
nvcc --version
cmake --version
gcc --version
free -h
df -h
```

Expected architecture:

```text
aarch64
```

Confirm CUDA is available:

```bash
ls -l /usr/local/cuda
ls -l /usr/local/cuda/bin/nvcc
```

For Jetson AGX Orin, the CUDA compute architecture is:

```text
87
```

---

## 2. Install build dependencies

```bash
sudo apt update

sudo apt install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libcurl4-openssl-dev
```

Optional HTTPS support for the embedded `llama.cpp` web server:

```bash
sudo apt install -y libssl-dev
```

Verify the tools:

```bash
git --version
ninja --version
pkg-config --version
```

---

## 3. Create the local LLM workspace

```bash
mkdir -p ~/local_llm
cd ~/local_llm
```

Recommended folder structure:

```text
~/local_llm/
├── llama.cpp/
└── models/
    └── Qwen3.5-9B/
```

---

## 4. Clone `llama.cpp`

```bash
cd ~/local_llm

git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

Record the exact source revision:

```bash
git rev-parse HEAD
git log -1 --oneline
```

After a successful build, preserve the working commit:

```bash
git rev-parse HEAD > llama_cpp_working_commit.txt
```

Do not run `git pull` again until the model has been tested successfully.

---

## 5. Configure the CUDA build

From the `llama.cpp` directory:

```bash
cd ~/local_llm/llama.cpp
rm -rf build

cmake -S . -B build \
    -G Ninja \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_BUILD_TYPE=Release
```

Important successful configuration messages include:

```text
CUDA Toolkit found
The CUDA compiler identification is NVIDIA
Using CMAKE_CUDA_ARCHITECTURES=87
Including CUDA backend
Configuring done
Generating done
```

An NCCL warning is acceptable for a single Jetson GPU.

---

## 6. Build `llama.cpp`

```bash
cd ~/local_llm/llama.cpp
cmake --build build --parallel 6
```

If the Jetson becomes unresponsive or memory pressure is high, reduce the number of jobs:

```bash
cmake --build build --parallel 2
```

Check the binaries:

```bash
ls -lh build/bin/
```

Important executable include:

```text
llama-cli
llama-server
llama-bench
llama-quantize
```

Verify the build:

```bash
./build/bin/llama-cli --version
./build/bin/llama-server --version
```

Confirm that the CUDA library exists:

```bash
ls -lh build/bin/libggml-cuda.so*
```

---

## 7. Install the Hugging Face download utility

Create and activate a Python environment if desired, then install the packages listed in `requirements-local-llm.txt`:

```bash
python3 -m pip install --user -r requirements-local-llm.txt
```

Verify the Hugging Face CLI:

```bash
~/.local/bin/hf --help
```

A Hugging Face login is usually not required for this public model repository.

---

## 8. Download Qwen3.5-9B Q4_K_M

Create the model directory:

```bash
mkdir -p ~/local_llm/models/Qwen3.5-9B
cd ~/local_llm/models/Qwen3.5-9B
```

Download only the selected quantized GGUF file:

```bash
~/.local/bin/hf download \
    unsloth/Qwen3.5-9B-GGUF \
    Qwen3.5-9B-Q4_K_M.gguf \
    --local-dir .
```

Verify the file:

```bash
ls -lh ~/local_llm/models/Qwen3.5-9B/
du -h ~/local_llm/models/Qwen3.5-9B/Qwen3.5-9B-Q4_K_M.gguf
```

Expected file size is approximately 5.7 GB.

Do not clone the complete Hugging Face repository because it contains many quantization variants.

---

## 9. Test the model with `llama-cli`

```bash
cd ~/local_llm/llama.cpp

./build/bin/llama-cli \
    --model ~/local_llm/models/Qwen3.5-9B/Qwen3.5-9B-Q4_K_M.gguf \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
    --threads 6 \
    --prompt "In two sentences, explain what a graphical user interface is." \
    --n-predict 300
```

### Important parameters

- `--ctx-size 8192`
  - Maximum working context for system instructions, retrieved documentation, conversation history, user input, and generated output.
- `--n-gpu-layers 99`
  - Requests that all available model layers be offloaded to CUDA.
- `--threads 6`
  - Number of CPU threads used for CPU-side work.
- `--n-predict 300`
  - Maximum number of newly generated tokens.

---

## 10. Verify GPU offloading

Run a verbose test:

```bash
cd ~/local_llm/llama.cpp

./build/bin/llama-cli \
    --model ~/local_llm/models/Qwen3.5-9B/Qwen3.5-9B-Q4_K_M.gguf \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
    --threads 6 \
    --prompt "Write one sentence describing a hyperspectral camera." \
    --n-predict 100 \
    --verbose 2>&1 | tee ~/local_llm/qwen_gpu_test.log
```

Search the log:

```bash
grep -Ei "CUDA|offload|GPU|device" ~/local_llm/qwen_gpu_test.log
```

Successful GPU output should contain lines similar to:

```text
CUDA0 : Orin
CUDA : ARCHS = 870
using device CUDA0 (Orin)
layer 0 assigned to device CUDA0
```

Monitor the Jetson during inference:

```bash
sudo tegrastats
```

---

## 11. Start `llama-server`

For the PANCAM GUI Help Agent, thinking mode is disabled so the model returns a concise final answer instead of spending the output budget on reasoning text.

```bash
cd ~/local_llm/llama.cpp

./build/bin/llama-server \
    --model ~/local_llm/models/Qwen3.5-9B/Qwen3.5-9B-Q4_K_M.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
    --threads 6 \
    --parallel 1 \
    --chat-template-kwargs '{"enable_thinking":false}'
```

Expected output:

```text
model loaded
listening on http://127.0.0.1:8080
```

Keep this terminal open while using the model.

Use `127.0.0.1` rather than `0.0.0.0` so the server is accessible only from the Jetson itself.

---

## 12. Test the health endpoint

Open another terminal:

```bash
curl http://127.0.0.1:8080/health
```

Expected response:

```json
{"status":"ok"}
```

---

## 13. Test the OpenAI-compatible chat endpoint

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B",
    "messages": [
      {
        "role": "system",
        "content": "You are the PANCAM GUI Help Agent. Give clear, concise answers. Do not invent PANCAM software behavior."
      },
      {
        "role": "user",
        "content": "What is the purpose of a GUI help assistant?"
      }
    ],
    "temperature": 0.2,
    "max_tokens": 200,
    "stream": false
  }'
```

A successful response contains:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      }
    }
  ]
}
```

The PANCAM software should read:

```text
choices[0].message.content
```

---

## 14. Recommended initial runtime settings

```text
Model:            Qwen3.5-9B-Q4_K_M
Context size:     8192 tokens
GPU layers:       99
CPU threads:      6
Parallel slots:   1
Thinking mode:    disabled
Temperature:      0.2
Maximum output:   200-600 tokens
Server address:   http://127.0.0.1:8080
```

---

## 15. Optional web interface

While `llama-server` is running, open:

```text
http://127.0.0.1:8080
```

The embedded `llama.cpp` web interface can be used to test the model before connecting it to PANCAM.

---

## 16. Stopping the server

In the terminal running `llama-server`, press:

```text
Ctrl+C
```

---

## 17. Updating `llama.cpp` later

First record the current working commit:

```bash
cd ~/local_llm/llama.cpp
git rev-parse HEAD
```

To update:

```bash
cd ~/local_llm/llama.cpp
git pull
rm -rf build

cmake -S . -B build \
    -G Ninja \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel 6
```

If the new build causes problems, return to the previously saved commit:

```bash
git checkout <WORKING_COMMIT_HASH>
rm -rf build
```

Then rebuild using the same CMake commands.

---

## 18. Troubleshooting

### `nvcc` is not found

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CUDA was not included in the build

Reconfigure with:

```bash
-DGGML_CUDA=ON
```

Confirm that this file exists:

```bash
ls build/bin/libggml-cuda.so*
```

### The model answer is empty and only `reasoning_content` is returned

Restart the server with:

```bash
--chat-template-kwargs '{"enable_thinking":false}'
```

### The answer stops early

Increase:

```text
max_tokens
```

or, in `llama-cli`:

```bash
--n-predict 500
```

### The server is not reachable

Check:

```bash
curl http://127.0.0.1:8080/health
```

Confirm that `llama-server` is still running in another terminal.

### The Jetson becomes slow

Reduce build or runtime resource use:

```text
Build jobs:       6 -> 2
Context size:  8192 -> 4096
Parallel slots:   keep at 1
```

Monitor memory and GPU load:

```bash
free -h
sudo tegrastats
```

---

## 19. PANCAM integration stage

After the local server is verified, the PANCAM AI architecture will be developed as:

```text
AI sidebar
    -> AI worker thread
    -> Agent manager
    -> GUI Help Agent
    -> Markdown document retriever
    -> Local backend client
    -> llama-server
    -> Qwen3.5-9B
```
