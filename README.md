# Blended RAG: Improving RAG Accuracy with Semantic Search & Hybrid Queries
Code base for the paper "Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers"
Paper Link- 

### Abstract
Retrieval-Augmented Generation (RAG) is a prevalent approach to infuse a private knowledge base of documents with Large Language Models (LLM) to build Generative Q\&A (Question-Answering) systems. However, RAG accuracy becomes increasingly challenging as the corpus of documents scales up; with Retrievers playing an outsized role in the overall RAG accuracy by extracting the most relevant document from the corpus to provide context to the LLM. In this paper, we propose the method of leveraging semantic search techniques such as Dense Vector indexes and Sparse Encoder Based indexes, blended with hybrid query strategies. Our study achieve better retrieval results and sets new benchmarks for IR (Information Retrieval) datasets like NQ and TREC-COVID datasets. We further extend such a 'Blended Retriever' to RAG system, to demonstrate far superior results on Generative Q\&A datasets like SQUAD, even surpassing fine-tuning performance.

## Scheme of Creating Blended Retrievers using Semantic Search with Hybrid Queries
<img src ="image/image1.png" />


## Results 
### 1. Retriever Accuracy 
The following section encapsulates the retrieval accuracy of our evaluative approach, quantified by Top-k metrics where \( k \in \{5, 10, 20\} \), across various datasets:
\begin{enumerate}
    \item NQ (Natural Questions) dataset
    \item TREC-Covid dataset
    \item SQuAD (Stanford Question Answering Dataset)
    \item CoQA (Conversational Question Answering)
\end{enumerate}


##### Top-5 Accuracy
| Top-5 retrieval accuracy | BM25 + Match Query | BM25+ Best Field | KNN + Match Query | KNN + Best Field | SERM + Match Query | SERM + Best Field |
| ------------------------ | ------------------ | ---------------- | ----------------- | ---------------- | ------------------- | ------------------ |
| NQ Dataset               | 25.19              | 85.05            | 87                | 87.67            | 88                  | 88.22              |
| Trec-covid Score1        | 36                 | 40               | 36                | 40               | 46                  | 48                 |
| Trec-covid Score2        | 86                 | 86               | 86                | 92               | 92                  | 94                 |                   |                    |
| SqUAD                    | 91.5               | 91.52            | 94.86             | 94.89            | 90.7                | 90.7               |

<img src ="image/image9.png">

##### Top-10 Accuracy
| Top-10 retrieval accuracy | BM25 + Match Query | BM25+ Best Field | KNN + Match Query | KNN + Best Field | SERM + Match Query | SERM + Best Field |
| ------------------------- | ------------------ | ---------------- | ----------------- | ---------------- | ------------------- | ------------------ |
| NQ Dataset                | 36.7               | 86.26            | 88.46             | 88.66            | 88.55               | 88.77              |
| Trec-covid Score1         | 66                 | 72               | 66                | 74               | 52                  | 78                 |
| Trec-covid Score2         | 92                 | 96               | 96                | 97               | 64                  | 98                 |
| SqUAD                     | 94.43              | 94.49            | 97.43             | 97.43            | 94.13               | 94.16              |


<img src ="image/image10.png">

##### Top-20 Accuracy

| Top-20 retrieval accuracy | BM25 + Match Query | BM25+ Best Field | KNN + Match Query | KNN + Best Field | SERM + Match Query | SERM + Best Field |
| ------------------------- | ------------------ | ---------------- | ----------------- | ---------------- | ------------------- | ------------------ |
| NQ Dataset                | 37.13              | 87.12            | 88.58             | 88.66            | 88.66               | 88.88              |
| Trec-covid Score1         | 86                 | 90               | 90                | 92               | 94                  | 98                 |
| Trec-covid Score2         | 98                 | 100              | 100               | 100              | 100                 | 100                |
| SqUAD                     | 96.3               | 96.36            | 98.57             | 98.58            | 96.49               | 96.52              |


<img src ="image/image11.png">

It can be concluded from the results that 'Blended Retriever' offers better accuracy than current methods across all the datasets. Sparse EncodeR Retriever Model (SERM) Based index with Best field queries often given best results with 88\% top-5 accuracy for NQ-Dataset and 94\% on TREC-Covid. The numbers increase for Top-10 and Top-20 accuracy. Below tables show all these results.

### Retriever Benchmarking using NDCG@10 Metric

| Dataset    | Model/Pipeline | NDCG@10 |
|------------|----------------|---------|
| Trec-covid | COCO-DR Large  | 0.804   |
| Trec-covid | Blended RAG    | 0.87    |
| NQ dataset | monoT5-3B      | 0.633   |
| NQ dataset | Blended RAG    | 0.67    |

Blended RAG performs better than the existing benchmark on both Trec-covid and NQ dataset.

### 2.  Blended RAG Accuracy

Distinctively, our Blended RAG approach has not undergone training on any related corpora. It harnesses an optimized amalgamation of field selections, query formulations, indices, and Large Language Models (LLMs) to render the most precise responses possible.  We used FlanT5-XXL(11B) for this pipeline. Consequently, the Blended RAG showcases enhanced performance in the RAG use case, even without dataset-specific fine-tuning. This characteristic renders it particularly advantageous for large enterprise datasets, where fine-tuning may be impractical or unfeasible, underscoring this research's principal application. 

##### 2.1  NQ Dataset

Evaluation of various queries for RAG pipeline on the NQ dataset.

| Query Types         | EM    | F1    | blue_score | meteor_score | rouge_score | sentence_similarity | sim_hash | perplexity_score | bleurt_score1 | bert_score |
| ------------------- | ----- | ----- | ---------- | ------------ | ----------- | ------------------- | -------- | ---------------- | ------------- | ---------- |
| BM25 + Match Query  | 32.91 | 40.4  | 3.81       | 33.47        | 42.65       | 57.47               | 18.95    | 3.15             | 27.73         | 6.11       |
| BM25+ BestField     | 37.58 | 47.31 | 4.63       | 3.98         | 49.79       | 63.33               | 17.02    | 3.07             | 13.62         | 65.11      |
| KNN + Match Query   | 40.21 | 50.51 | 4.77       | 42.11        | 53.32       | 67.02               | 15.94    | 3.04             | 5.12          | 67.27      |
| KNN + Best Field    | 40.32 | 50.45 | 5.05       | 42.34        | 53.24       | 66.88               | 15.94    | 3.048            | 5.7           | 67.3       |
| SERM + Match Query | 42.63 | 53.96 | 5.27       | 45.13        | 57.07       | 70.47               | 14.95    | 3.01             | 2.02          | 69.25      |
| SERM + BestField   | 42.3  | 53.25 | 5.24       | 44.77        | 56.36       | 69.65               | 15.14    | 3.02             | 0.24          | 68.97      |

#### 2.2 Squad Dataset

##### Evaluation of the RAG Pipeline on the SquAD dataset.

| Model/Pipeline | EM     | F1    | Top-5 | Top-20 |
|----------------|--------|-------|-------|--------|
| RAG-original   | 28.12  | 39.42 | 59.64 | 72.38  |
| RAG-end2end    | 40.02  | 52.63 | 75.79 | 85.57  |
| Blended RAG    | 57.63  | 68.4  | 94.89 | 98.58  |

Blended RAG outperforms the existing RAG systems.


###### Evaluation of various queries for RAG pipeline on the SquAD dataset.


| Query Types         | EM    | F1    |
| ------------------- | ----- | ----- |
| BM25 + Match Query  | 56.07 | 67.12 |
| BM25 + Best Field   | 56.07 | 67.12 |
| SERM + Match Query | 54.92 | 65.75 |
| SERM + Best Field  | 54.96 | 65.79 |
| KNN + Match Query   | 57.63 | 68.4  |
| KNN + Best Field    | 57.63 | 68.4  |


### Code
- **evaluation_blended_rag.py**: This script is used to evaluate the Blended RAG pipeline and generate the results.

- **evaluation_charts.py**: The evaluation results were generated by `evaluation_retrieval.py` for retriever and `evaluation_blended_rag.py` for the Blended RAG pipeline. This notebook uses those results to generate the charts published in the paper.

- **evaluation_retrieval.py**: This script is used to evaluate the retrieval methods and generate the results.

- **indexing.py**: This script is used to create the index.


### Input 
This module uses various inputs, such as mapping and search_query, to index and search the queries at the index.

- **mapping/**: Contains mapping files with respective BM25, KNN and Sparse_Encoder.
- **search_query/**: A collection of search_queries used across different evaluation tasks.



# Disclaimer
 The content may include systems & methods pending patent with the USPTO and protected under US Patent Laws. In case of any questions, please reach out to kunal@ibm.com.
