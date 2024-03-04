import json
from datasets import Dataset
import pandas as pd
import numpy as np
import warnings
from transformers import AutoTokenizer
from transformers import AutoModel
from datasets import load_dataset
import torch
import faiss
import pickle
warnings.filterwarnings("ignore")

file_path = 'data/tokenized_dataset.json'
max_records = 100000
processed_records = 0

model = AutoModel.from_pretrained('stanford-crfm/BioMedLM')
tokenizer = AutoTokenizer.from_pretrained('stanford-crfm/BioMedLM')

data = {
    #'article': [],
    'abstract': [],
    #'section_names': [],
    #'article_tokenized_chunks': [],
    'abstract_tokenized_chunks': [],
    #'section_names_tokenized_chunks': []
}

try:
    with open(file_path, 'r') as file:
        for line in file:
            try:
                record = json.loads(line)

                # Append data for each feature
                #data['article'].append(record.get('article', ''))
                data['abstract'].append(record.get('abstract', ''))
                #data['section_names'].append(record.get('section_names', []))
                #data['article_tokenized_chunks'].append(record.get('article_tokenized_chunks', []))
                data['abstract_tokenized_chunks'].append(record.get('abstract_tokenized_chunks', []))
                #data['section_names_tokenized_chunks'].append(record.get('section_names_tokenized_chunks', []))

                processed_records += 1
                if processed_records >= max_records:
                    break
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line: {processed_records + 1}")
                print(str(e))
                break  # or use `continue` to skip this record

except FileNotFoundError:
    print(f"The file {file_path} was not found.")

dataset = Dataset.from_dict(data)

def convert_to_token_ids(chunks):
    """Convert tokenized text chunks to token IDs, ensuring they are within the model's range."""
    token_ids = []
    for chunk in chunks:
        ids = tokenizer.convert_tokens_to_ids(chunk)
        ids = [id_ for id_ in ids if id_ < tokenizer.vocab_size]
        token_ids.extend(ids)
    return token_ids

def get_embedding(token_ids):
    """Generate embedding for token IDs, handling sequences longer than the model's max length."""
    max_length = 1024 
    chunks = [token_ids[i:i + max_length] for i in range(0, len(token_ids), max_length)]

    embeddings = []
    with torch.no_grad():
        for chunk in chunks:
            if len(chunk) > 0:
                inputs = torch.tensor(chunk).unsqueeze(0)
                if inputs.size(1) > 0:
                    outputs = model(inputs)
                    chunk_embedding = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(chunk_embedding)

    if len(embeddings) == 0:
        return np.zeros(model.config.hidden_size) 

    embeddings = torch.stack(embeddings).mean(dim=0)
    return embeddings.squeeze().numpy()

from tqdm import tqdm  

embeddings = []
abstract_map = {}

for i, row in tqdm(enumerate(dataset), total=len(dataset), desc="Processing"):
    token_ids = convert_to_token_ids(row['abstract_tokenized_chunks'])
    embedding = get_embedding(token_ids)
    embeddings.append(embedding)
    abstract_map[i] = row['abstract']

embeddings = np.array(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

with open('abstract_map.pkl', 'wb') as pickle_file:
    pickle.dump(abstract_map, pickle_file)

faiss.write_index(index, 'KB_100k_abstracts_[date].faiss')