import os
import re
from datetime import date
import pandas as pd
import json
from datetime import datetime
import requests

from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
import warnings
warnings.filterwarnings('ignore')

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


## Search in BM25 index and ELSER index
def processESIndex(df_questions,search_query,index_name):
    for ind in df_questions.index:
        print("Processsing -----",ind)
        question =df_questions['text'][ind]
        response = es_client.search(
        index=index_name,
        body=search_query,
        scroll='5m',  # Set the scroll timeout (e.g., 5 minutes)
        size=10  # Set the number of documents to retrieve per scroll
        )
        all_hits = response['hits']['hits']
        print(len(all_hits))
        flag = False
        for num, doc in enumerate(all_hits):
            if df_questions['answers'][ind] in  doc["_source"]['content']:
               # print("helooo",doc["_source"]['content_para'])
                #print ("DOC ID:", doc["_id"], "--->", doc, type(doc), "\n")
                flag = True
                break
        print ("DOC Score:", flag)
        df_questions['model_op1'][ind] = flag
            
    return df_questions


## search in KNN index
def processESIndex_Knn(df_questions,search_query,index_name):
    i =0
    count =0
    for ind in df_questions.index:
        print("Processsing -----",ind)
        question =df_questions['text'][ind]
        content_embedding =model.encode(question)
        ## content_embedding will be add into your query according the question
        response = es_client.search(
        index=index_name,
        body=search_query,
        scroll='5m',  # Set the scroll timeout (e.g., 5 minutes)
        size=10  # Set the number of documents to retrieve per scroll
        )
        all_hits = response['hits']['hits']
        print(len(all_hits))
        flag = False
        for num, doc in enumerate(all_hits):
            if df_questions['answers'][ind] in  doc["_source"]['content']:
                flag = True
                break
        print ("DOC Score:", flag)
        df_questions['model_op_kNN'][ind] = flag
    return df_questions


## Read the file
search_query=""
with open('./input/search_query/BM25/bm25_best.txt', 'r') as file:
    search_query = file.read().rstrip()

## If you need to replace any file or  other query basis of Your index you can use the respective folders
index_name ="research_index_bm25"
df_questions =pd.read_csv("input question file")

## BM25 and ELSER
processESIndex(df_questions,search_query,index_name)

## KNN
processESIndex_Knn(df_questions,search_query,index_name)
