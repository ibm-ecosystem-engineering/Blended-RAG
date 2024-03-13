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


es_new = Elasticsearch("https://elastic:fR1u37hSTV2r5w20UHb03TQ8@elastic-elastic-8-10.htalukder-sap-testing-2bef1f4b4097001da9502000c44fc2b2-0000.ca-tor.containers.appdomain.cloud",  ca_certs=False,
                   verify_certs=False)
es_new.info()

## Download model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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


def nq_daatset_para(df_docs,source):
    i=0
    for index, row in df_docs.iterrows():
            i=i+1
            print("Processing i",index)
            id_ = row['id']
            text = row['text']
            title = row['title']
            source = source
            text = pre_processingtext(text)

            values = text.split(" ")
            len_values = len(values)
            para = ''
            token_num = int(len_values/512)+1
            content_embedding = model.encode(text)
            for k in range(token_num):
                if k == 0:
                    start =0
                    end = 512
                else:
                    if k == 1:
                        start = end 
                        end = start+k*512
                    else:
                        start = end 
                        end = end+512
                para =''
                if start == end:
                    end = len_values
                    print(k,start,end)
                if end > len_values:
                    end = len_values
                for i in range(start,end):
                    para = para+" "+values[i]
                
                    id_val = title+"_"+str(k)
                    content_para_embedding = model.encode(para)
                    doc ={
                                "id": ""+id_val+"",
                                "source": ""+source+"",
                                "content": ""+text+"",
                                "content_para": ""+para+"",
                                "title": ""+title+"",
                                "metadata": ""
                            }
                    
                    doc1 ={
                                "id": ""+id_val+"",
                                "source": ""+source+"",
                                "content": ""+text+"",
                                "content_para": ""+para+"",
                                "content_embedding": content_embedding,
                                "content_para_embedding": content_para_embedding,
                                "title": ""+title+"",
                                "metadata": ""
                            }
                    

                    response = es_new.index(index=index_name_bm25, body=doc)
                    print(response)
                    response = es_new.index(index=index_name_knn, body=doc1)
                    print(response)

index_name_knn = 'research_index_knn_para'
index_name_bm25 = "research_index_para_bm25"

doc_folder_msmarco = '/Users/abhilashamangal/Documents/Semantic Search/data/msmarco/'
files_msmarco = get_all_files(doc_folder_msmarco)
df_corpus = pd.read_json(files_msmarco[2],lines=True)

print("Helooo my time---")
source ="msmarco"
nq_daatset_para(df_corpus,source)