import os
import optuna

from sklearn.metrics import mean_squared_error
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

import config as C
import utils as U


def similarity_ranking(person_embeddings: dict, name: str) -> list[list]:
    """
    Given the embeddings of people's description and a person, return a list of cosine similarities
    and cosine similarity rankings of that person to other people, sorted alphabetically by name.

    Parameters
    ----------
    person_embeddings: {str: list[float]}
        dictionary {name: description_embedding}
    name: str
        name of the person from whom to calculate cosine similarities with other people

    Return
    ------
    top_matches: list[list[str, float, int]]
        [[name, cosine_similarity, rank]] ranked alphabetically by name
    """
    # calculate cosine similarity
    top_matches = []
    for person in person_embeddings.keys():
        top_matches.append([person, util.cos_sim(person_embeddings[name], person_embeddings[person]).item()])

    # add similarity ranking
    top_matches.sort(key=lambda x: -x[1])  # sort by cosine similarity
    for i, match in enumerate(top_matches):
        match.append(i)

    # # sort by name
    top_matches = top_matches[1:]
    top_matches.sort(key=lambda x: x[0])
    return top_matches


def get_all_ranks(person_embeddings: dict) -> list:
    """
    Given the embeddings of people's description, for each person, get a list of
    cosine similarity rankings of other people (sorted alphabetically) by name to that person.
    Return the concanation of all those lists.

    Parameters
    ----------
    person_embeddings: {str: list[float]}
        dictionary {name: embedding}

    Return
    ------
    rank: list[int]
    """
    rank = []

    for target_person in person_embeddings.keys():
        top_matches = similarity_ranking(person_embeddings, target_person)
        rank += [top_matches[i][2] for i in range(len(top_matches))]

    return rank


class objective:
    def __init__(self, file_name, 
                        model_src, 
                        dim_components, 
                        random_state):
        
        ## Getting data ready
        self.data_src = os.path.join(C.BASE_DATA_PATH, file_name)
        self.mapped_data = U.read_data_from_csv(self.data_src)
        ## Loading Model
        self.model = SentenceTransformer(model_src)
        ## Creating embeddings
        paragraphs = list(self.mapped_data.keys())
        self.embeddings = self.model.encode(paragraphs)
            
        ## Scaling Data
        scaler = StandardScaler()
        self.scaled_embeddings = scaler.fit_transform(list(self.embeddings))    
        
        ## Create a dictionary to store embeddings for each person
        self.person_embeddings = {self.mapped_data[paragraph]: embedding for paragraph, embedding in zip(paragraphs, self.scaled_embeddings)}
        
        ## Ranking all
        self.rank_true = get_all_ranks(self.person_embeddings)
        self.random_state = random_state

    def __call__(self, trial):
        n_neighbors = trial.suggest_int("n_neighbors", 2, 53)
        min_dist = trial.suggest_float("min_dist", 0, 0.99)
        ## Dimensionality Reduction
        reducer = umap.UMAP(n_neighbors = n_neighbors
                            ,min_dist = min_dist
                            ,n_components = 2
                            ,transform_seed = self.random_state
                            ,random_state=self.random_state)
        
        reduced_data = reducer.fit_transform(self.embeddings)
        
        pupils = list(self.person_embeddings.keys())
        # Create a dictionary to store embeddings for each person
        person_red_embeddings = {user: embedding for user, embedding in zip(pupils, reduced_data)}

        ## Ranking Reduced All
        rank_pred = get_all_ranks(person_red_embeddings)
        mse = mean_squared_error(self.rank_true, rank_pred)
        return mse


def hyper_param_search_routine(file_name, 
                                model_src, 
                                dim_components, 
                                random_state):

    

    study = optuna.create_study()
    study.optimize(objective(file_name, 
                                model_src, 
                                dim_components, 
                                random_state), n_trials=100)
    
    trial = study.best_trial
    print("Best Score: ", trial.value)
    print("Best Params: ")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))
        
        

    