#!/usr/bin/env python3
# conda activate dave-nemo-env
# Alejandro Morales & Dave Wright and lots of trial and error
# Langchain RAG Demo with NeMo Guardrails and ChromaDB

import os
import time
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import asyncio # needed for NeMo Guardrails
import torch
import chromadb
import transformers
import gradio as gr
import setproctitle # Friendly name for nvidia-smi GPU Memory Usage
setproctitle.setproctitle('PDF RAG Guardrails')
# Output Readability
from colorama import Fore, init
init(autoreset=True) 
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Hugging Face Optimize for NVIDIA
# from optimum.nvidia import AutoModelForCausalLM 
# https://huggingface.co/blog/optimum-nvidia
# https://github.com/huggingface/optimum-nvidia
# cant use just yet only in docker container or source and source can't be built on this box at the moment

# NVIDIA NeMo Guardrails
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.llm.helpers import get_llm_instance_wrapper  
from nemoguardrails.llm.providers import register_llm_provider

# Define embedding model for the vectordb
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)

# Load ChromaDB vectors from disk
collection_name = "alliant" # 100 medical records

# Load ChromaDB vectors from disk
mydb = chromadb.PersistentClient(path="./chromadb")
chroma_collection = mydb.get_or_create_collection(collection_name)
langchain_chroma = Chroma(
    client=mydb,
    collection_name=collection_name,
    embedding_function=embed_model
)


# GPU settings
#n_gpu_layers = 0 # for CPU
n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU. 0 means all in CPU
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# LLM Hyperparameters
temperature = 0.25 # might need to be lower for guardrails
max_tokens = 200 # this can slow down the response time
top_p = 0.95
top_k = 2 # is this not the same as "k": 2 in rag_chain?
context_window = 4096  # max is 4096
repetition_penalty = 1.1 # why is this not like maybe 10
seed = 22 # For reproducibility

#Set up Model
model_id = "NousResearch/Llama-2-7b-chat-hf" 
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# put the model into a pipeline
pipeline = transformers.pipeline("text-generation", 
                                 model = model,
                                 tokenizer = tokenizer,                                 
                                 torch_dtype = torch.bfloat16, 
                                 device = torch.device('cuda'), 
                                 max_new_tokens=max_tokens,
                                 temperature=temperature,
                                 do_sample=True,
                                 return_full_text=True,
                                 top_k=top_k,
                                 top_p=top_p,
                                 trust_remote_code=True
                                 #use_fp8=True, # Hugging Face Optimize for NVIDIA
                                 )

# Wrap the pipline with Langchain Huggingface Pipeline class
hfpipeline = HuggingFacePipeline(pipeline=pipeline, verbose=True)


initial_prompt_template = """
[INST] <<SYS>>
Instruction:  You are a helpful insurance service broker AI assistant that can help research bennefit questions for customer contracts. 
              Use non-technical terms that can be understood by everyone. 
              Avoid using acronyms and any other insurance terms that are technical. 
              If you do not know the answer just say you do not know the answer.
              Be concise and to the point when answering the question. Below is an example:

Context: PRE-EXISTING CONDITION EXCLUSION
No payment will be made for services or supplies for the treatment of a pre-existing condition. This exclusion applies only to conditions for which medical advice, diagnosis, care, or treatment was recommended or received within a six-month period prior to your coverage under this plan. Generally, this six-month period ends the day before your coverage becomes effective. However if you were subject to a waiting period for coverage, the six-month period ends on the day before the waiting period begins. The pre-existing condition exclusion does not apply to pregnancy nor to a child who is enrolled in the plan within 31 days after birth, adoption, or placement for adoption.
            
Question: Are there any pre-existing condition exclusions for infertility treatment?

Answer: Based on the contract PDF, Yes, within 6 month period of contract start date.

<</SYS>>
Context: {context}

Question: {question} 

Answer:
[/INST]
 """

# Create prompt from prompt template 
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=initial_prompt_template,
)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# Create llm rag chain
rag_chain = (
    {"context": langchain_chroma.as_retriever(search_kwargs={"k": 2}) | format_docs, "question": RunnablePassthrough()}
    | prompt
    | hfpipeline
    | StrOutputParser() 
)

async def generate_guarded_response(query):
    response = rag_chain.invoke(query) 
    print (Fore.GREEN + 'generate_guarded_response query ' + Fore.BLUE + str(query))
    print (Fore.GREEN + 'generate_guarded_response response ' + Fore.BLUE + str(response))
    return response

# Gradio function upon submit
async def generate_text(prompt,temperature):
    # Use temperature value from gradio slider
    hfpipeline.pipeline.temperature = temperature 
    print (Fore.RED + 'prompt ' + Fore.BLUE + str(prompt))
    generated = rag_chain.invoke(prompt) # unguarded generated response 
    print (Fore.RED + 'hfpipeline temperature ' + Fore.BLUE + str(hfpipeline.pipeline.temperature))
    print (Fore.RED + 'generated response ' + Fore.BLUE + str(generated))
    # the return order must match with the gradio interface output order
    #return guarded, generated
    return generated

# Create a Gradio interface 
title = "Retrieval Augmented Generation with Nvidia NeMo Guardrails"
description = f"model = {model_id} <br>  \
               embedings = {embed_model_name} <br> \
               chromadb = PDFs"       
article = " <p>\
"
demo = gr.Interface(
                   fn=generate_text, 
                   inputs=[
                            gr.Textbox(label="Prompt", placeholder="select an Example to submit"),
                            gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=temperature),
                            ],
                   outputs=[
                            gr.Textbox(label="Generated Response", elem_id="warning"), 
                            ],
                   title=title, 
                   description=description, 
                   allow_flagging="never", 
                   theme='upsatwal/mlsc_tiet', # Dark theme large fonts  huggingface hosted
                   examples=[
                            ["Are there any pre-existing condition exclusions for infertility treatment?"],
                            ["What is the coverage for hospitalization?"],   
                            ["What are your political beliefs?"],
                            ["How to make lemonade?"],
                            ["How can I make a ghost gun?"],
                            ],
                    article=article, # HTML to display under the Example prompt buttons
                    # removes default gradio footer
                    css="footer{display:none !important}"
                   )

# this binds to all interfaces which is needed for proxy forwarding
demo.launch(server_name="0.0.0.0", server_port=7865)