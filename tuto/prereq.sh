pip install llama-index
pip install sentence-transformers

# GPU
#!CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
# CPU
CMAKE_ARGS="-DLLAMA_CUBLAS=off" pip install llama-cpp-python

pip install langchain

if [ ! -f vigogne-2-7b-chat.Q5_K_M.gguf ]
then 
    wget https://huggingface.co/TheBloke/Vigogne-2-7B-Chat-GGUF/resolve/main/vigogne-2-7b-chat.Q5_K_M.gguf
fi
