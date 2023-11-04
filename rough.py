
import json
from scipy.spatial.distance import cosine



# def compare_embedding(str1: str, str2: str) -> float:
#     """
#     Given two strings, compare their embeddings' cosine similarity.
#     """


#     cosine_similarity = 1 - cosine(embedding_1, embedding_2)
#     return cosine_similarity



if __name__ == "__main__":
    path = 'analysis/mod_person_embeddings.json'
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
        try:
            candidate_cosine_similarity[candidate] = round((1 - cosine(candidate_embeddings[0], candidate_embeddings[1]))*100, 2)
        except Exception as ex:
            print(f"Candidate: {candidate} has no embeddings")
            print(f"Error: {str(ex)}")
            
    print(candidates)    
    print(candidate_cosine_similarity)
    