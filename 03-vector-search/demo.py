import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

import json
with open('documents.json') as f_in:
    docs_raw = json.load(f_in)

# print(docs_raw)

documents = []

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)

# print(documents[1])


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

# print(len(model.encode("This is a simple sentence")))

# operations = []
# for doc in documents:
#     # Transforming the title into an embedding using the model
#     doc["text_vector"] = model.encode(doc["text"]).tolist()
#     operations.append(doc)

# print(operations[1])

from elasticsearch import Elasticsearch
es_client = Elasticsearch('http://localhost:9200') 

print(es_client.info())

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
            "course": {"type": "keyword"} ,
            "text_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
        }
    }
}

index_name = "course-questions"

# es_client.indices.delete(index=index_name, ignore_unavailable=True)
# es_client.indices.create(index=index_name, body=index_settings)

# for doc in operations:
#     try:
#         es_client.index(index=index_name, document=doc)
#     except Exception as e:
#         print(e)

search_term = "windows or mac?"
vector_search_term = model.encode(search_term)

query = {
    "field": "text_vector",
    "query_vector": vector_search_term,
    "k": 5,
    "num_candidates": 10000, 
}

res = es_client.search(
    index=index_name, 
    knn=query, 
    source=["text", "section", "question", "course"]
)
print(res["hits"]["hits"])

response = es_client.search(
    index=index_name,
    query={
        "match": {"section": "General course-related questions"},
    },
    knn=query,
    size=5
)

print(response["hits"]["hits"])