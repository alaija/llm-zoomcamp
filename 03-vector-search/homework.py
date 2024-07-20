from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")

user_question = "I just discovered the course. Can I still join it?"

v = embedding_model.encode(user_question)

# print(v[0]) # 0.07822262

import json
with open('./eval/documents-with-ids.json') as f_in:
    docs = json.load(f_in)

documents = []
for doc in docs:
    if doc['course'] == "machine-learning-zoomcamp":
        documents.append(doc)

# print(len(documents)) # 375

from tqdm import tqdm

embeddings = []
print('Computing embeddings...')
for doc in tqdm(documents):
    qa_text = f'{doc["question"]} {doc["text"]}'
    embedding = embedding_model.encode(qa_text).tolist()
    embeddings.append(embedding)

import numpy as np

X = np.array(embeddings)
# # print(X.shape) # (375, 768)

scores = X.dot(v)
# # print(max(scores)) # 0.65

class VectorSearchEngine():
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=5):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [self.documents[i] for i in idx]

def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    relevance_total = []
    print(f'Evaluating with {search_function.__qualname__}')
    for q in tqdm(ground_truth):
        doc_id = q['document']
        v_q = embedding_model.encode(q['question'])
        results = search_function(v_q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }


search_engine = VectorSearchEngine(documents=documents, embeddings=X)
# results = search_engine.search(v, num_results=5)
 
import pandas as pd

ground_truth_url = './eval/ground-truth-data.csv'

df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records')

result = evaluate(ground_truth, search_function=search_engine.search)
print(result) # {'hit_rate': 0.9398907103825137, 'mrr': 0.8516484517304189}

from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200') 

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
            "text_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

def elastic_search(field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {
            "term": {
                "course": course
            }
        }
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs

print('Building index...')
for doc in tqdm(documents):
    question = doc['question']
    text = doc['text']
    qt = question + ' ' + text
    doc['question_vector'] = embedding_model.encode(question)
    doc['text_vector'] = embedding_model.encode(text)
    doc['question_text_vector'] = embedding_model.encode(qt)
    es_client.index(index=index_name, document=doc)

result = evaluate(ground_truth, search_function=lambda v: elastic_search('question_text_vector', v, 'machine-learning-zoomcamp'))
print(result) # {'hit_rate': 0.9218579234972678, 'mrr': 0.8335974499089249}
