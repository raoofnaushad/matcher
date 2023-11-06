import json

from sentence_transformers import SentenceTransformer, util


def compare_modified_vector_embeddings():

    path = 'results/data_analysis_embeddings.json'
    data = json.load(open(path))
    candidates = []
    for person in data:
        if 'Modified' in person:
            candidates.append(person[:-9])
    
    candidate_cosine_similarity = {}
    for candidate in candidates:
        candidate_embeddings = []
        for person in data:            
            # print("---")
            if candidate in person:
                # print(candidate, person)
                candidate_embeddings.append(data[person])
          
        candidate_cosine_similarity[candidate] = round(util.cos_sim(candidate_embeddings[0], candidate_embeddings[1]).item()*100, 2)

    
    return candidate_cosine_similarity