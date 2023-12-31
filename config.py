import os

import warnings
warnings.simplefilter("ignore")

RANDOM_STATE = 42

MINILM_L6_V2 = 'sentence-transformers/all-MiniLM-L6-v2'
MPNET_BASE_V2 = 'sentence-transformers/all-mpnet-base-v2'

BASE_DATA_PATH = 'data'
BASE_RESULTS_PATH = 'results'





MOD_PLOT_PATH = 'mod_visualization.png'
CLASSMATES_DATA_PATH = os.path.join(BASE_DATA_PATH, 'classmates.csv')
CLASSMATES_MOD_DATA_PATH = os.path.join(BASE_DATA_PATH, 'classmates_modified.csv')
PERSON_EMBEDDINGS_PATH = os.path.join(BASE_RESULTS_PATH, 'person_embeddings.json')
MOD_PERSON_EMBEDDINGS_PATH = os.path.join(BASE_RESULTS_PATH, 'mod_person_embeddings.json')
MOD_MODEL_EMBEDDINGS_PATH = os.path.join(BASE_RESULTS_PATH, 'mod_model_embeddings.json')
