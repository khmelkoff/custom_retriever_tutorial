# Custom Retriever Tutorial
LangChain custom retriever tutorial with text preprocessing and benchmark

## Preparation
Install Ollama and load the model. Tested with gemma3:27b-it-qat, qwen3:14b, gemma4:31b on GPU 16GB
```
set OLLAMA_CONTEXT_LENGTH=16000
ollama pull gemma3:27b-it-qat
```

Create a Python environment (recommended python=3.11)
```
conda create --name custom_retriever python==3.11
conda activate custom_retriever
```

Install dependencies from requirements along with Jupyter
```
pip install -r requirements.txt
```

Start the jupyter notebook
```
jupyter notebook
```
