from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from openai import OpenAI
import ollama
import requests 
import tiktoken
import logging

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

index_name = "new-course-questions"
es_client = Elasticsearch('http://localhost:9200')

client = OpenAI()

enc = tiktoken.encoding_for_model("gpt-4o")

def get_data():
    docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    documents = []

    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)
    return documents


def create_index_if_needed():
    if not es_client.indices.exists(index=index_name):
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
                    "course": {"type": "keyword"} 
                }
            }
        }
        es_client.indices.create(index=index_name, body=index_settings)
        documents = get_data()
        for doc in documents:
            es_client.index(index=index_name, document=doc)


def elastic_search(query):
    search_query = {
        "size": 3,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "machine-learning-zoomcamp"
                    }
                }
            }
        }
    }
    response = es_client.search(index=index_name, body=search_query)
    result_docs = []
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    return result_docs


def build_prompt(query, search_results):
    context_template = """
Q: {question}
A: {text}
""".strip()
    
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()
    context = ""

    for doc in search_results:
        context = context + context_template.format(question=doc['question'], text=doc['text'] + "\n\n")

    return prompt_template.format(question=query, context=context).strip()


def openai(prompt):
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def local_llama(prompt):
    response = ollama.chat(
        model='llama3',
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def rag(query, llm):
    search_results = elastic_search(query)
    prompt = build_prompt(query, search_results=search_results).strip()
    logger.info(f"Prompt length: {len(prompt)}")
    in_tokens = enc.encode(prompt)
    logger.info(f"In tokens count: {len(in_tokens)}")
    answer = llm(prompt)
    out_tokens = enc.encode(answer)
    logger.info(f"Out tokens count: {len(out_tokens)}\n\n")

    return answer


create_index_if_needed()

query = "How do I execute a command in a running docker container?"

response = rag(query, local_llama)
print("Local Llama response:")
print(response)

response = rag(query, openai)
print("OpenAI response:")
print(response)