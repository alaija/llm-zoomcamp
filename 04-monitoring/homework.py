import pandas as pd
import os
from urllib.request import urlretrieve

file_path = 'results-gpt4o-mini.csv'

if not os.path.isfile(file_path):
    url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv?raw=1'
    urlretrieve(url, file_path)
    
df = pd.read_csv(file_path).iloc[:300]

print("Q1. Getting embeddings model")
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from urllib.request import urlretrieve

MODEL_NAME = "multi-qa-mpnet-base-dot-v1"

model = SentenceTransformer(MODEL_NAME)
answer_llm = df.iloc[0].answer_llm
embedding_answer_llm = model.encode(answer_llm)

print(embedding_answer_llm[0]) # -0.42244643

print("Q2. Computing the dot product")

from tqdm.auto import tqdm

def compute_similarity(record):
    answer_orig = record['answer_orig']
    answer_llm = record['answer_llm']
    
    v_llm = model.encode(answer_llm)
    v_orig = model.encode(answer_orig)
    
    return v_llm.dot(v_orig)

evaluations = []
for record in tqdm(df.to_dict(orient='records')):
    sim = compute_similarity(record)
    evaluations.append(sim)

df['dot_product'] = evaluations
print(df['dot_product'].describe()) # 75%       31.674307

print("Q3. Computing the cosine")
import numpy as np

def normalize(v):
    norm = np.sqrt((v * v).sum())
    v_norm = v / norm
    return v_norm

def compute_similarity_norm(record):
    answer_orig = record['answer_orig']
    answer_llm = record['answer_llm']
    
    v_llm = normalize(model.encode(answer_llm))
    v_orig = normalize(model.encode(answer_orig))

    return v_llm.dot(v_orig)

evaluations = []
for record in tqdm(df.to_dict(orient='records')):
    sim = compute_similarity_norm(record)
    evaluations.append(sim)

df['cosine'] = evaluations
print(df['cosine'].describe()) # 75%        0.836235

print("Q4. Rouge")
from rouge import Rouge
rouge_scorer = Rouge()

scores_list = []
for r in tqdm(df.to_dict(orient='records')):
    scores = rouge_scorer.get_scores(r['answer_llm'], r['answer_orig'])[0]
    scores_list.append(scores)

print(scores_list[10]['rouge-1']['f']) # 0.45454544954545456

print("Q5. Average Rouge")
print((scores_list[10]['rouge-1']['f'] + scores_list[10]['rouge-2']['f'] + scores_list[10]['rouge-l']['f'])/3) # 0.35490034990035496

print("Q6. Average Rouge for all records")
rouge2_scores = []
for scores in scores_list:
    rouge2_scores.append(scores['rouge-2']['f'])

print(sum(rouge2_scores) / len(rouge2_scores)) # 0.20696501983423318