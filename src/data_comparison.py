import json

from sentence_transformers import SentenceTransformer, util


def compare_modified_vector_embeddings(embeddings_path):

    data = json.load(open(embeddings_path))
    candidates = []
    for person in data:
        if 'Modified' in person:
            candidates.append(person[:-9])
    
    candidate_cosine_similarity = {}
    for candidate in candidates:
        candidate_embeddings = []
        for person in data:            
            if candidate in person:
                candidate_embeddings.append(data[person])
          
        candidate_cosine_similarity[candidate] = round(util.cos_sim(candidate_embeddings[0], candidate_embeddings[1]).item()*100, 2)

    
    return candidate_cosine_similarity