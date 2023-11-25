pip install llama-index
pip install sentence-transformers

# GPU
#!CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
# CPU
CMAKE_ARGS="-DLLAMA_CUBLAS=off" pip install llama-cpp-python

