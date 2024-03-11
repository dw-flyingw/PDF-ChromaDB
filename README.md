# PDF-ChromaDB

> [!IMPORTANT]
> This is currently a work in progress, expect things to be broken!

Demonstrate a Retrieval Augmented Generation on a locally running Llama-2-7b model with PDFs in a ChromaDB vector database.

Tested on HPE DL380, AMD CPU and Nvidia L40S GPU

# Install
huggingface-cli download NousResearch/Llama-2-7b-chat-hf

huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 

conda create nemo-env 

conda activate nemo-env

pip install -r requirements.txt 

python3 csv2chromadb.py

python3 main.py
