import os
import re
from datetime import date
import pandas as pd
import json
from datetime import datetime
import requests

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError



## Replace elastic instance here
es_client = Elasticsearch("https://esuser:espassword8@eshost:port",  ca_certs=False,
                   verify_certs=False)
es_client.info()

## Download model for KNN
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

## get the files from specific folder
def get_all_files(folder_name):
    # Change the directory
    os.chdir(folder_name)
    # iterate through all file
    file_path_list =[]
    for file in os.listdir():
        print(file)
        file_path = f"{folder_name}/{file}"
        file_path_list.append(file_path)
    return file_path_list


## create the index
def create_index(index_name,mapping):
    try:
        es_client.indices.create(index=index_name,body = mapping)
        print(f"Index '{index_name}' created successfully.")
    except RequestError as e:
        if e.error == 'resource_already_exists_exception':
            print(f"Index '{index_name}' already exists.")
        else:
            print(f"An error occurred while creating index '{index_name}': {e}")



def index_data(df_docs,source,index_name,index_name_knn):
    i=0
    for index, row in df_docs.iterrows():
        i=i+1
        print("Processing i",i)
        id_ = row['id']
        text = row['text']
        title = row['title']
        source = source
        text_embedding = model.encode(text)
        doc ={
                        "id": ""+title+"",
                        "source": ""+source+"",
                        "text_field": ""+text+"",
                        "title": ""+title+"",
                        "metadata": ""
            }
        doc_knn = {
                        "id": ""+title+"",
                        "source": ""+source+"",
                        "text": ""+text+"",
                        "title": ""+title+"",
                        "metadata": "",
                        "text_embedding": text_embedding
                    }
        response = es_client.index(index=index_name, body=doc)
        print(response)
        response = es_client.index(index=index_name_knn, body=doc_knn)
        print(response)           


## Example Index name
index_name_knn = 'research_index_knn'
index_name = "research_index_bm25"

doc_folder_msmarco = '/Users/abhilashamangal/Documents/Semantic Search/data/msmarco/'
files_msmarco = get_all_files(doc_folder_msmarco)
df_corpus = pd.read_json(files_msmarco[2],lines=True)

print("Helooo my time---")
source ="msmarco"
index_data(df_corpus,source)