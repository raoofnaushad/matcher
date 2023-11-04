from src import config as C

import csv
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from scipy.spatial.distance import cosine

import matplotlib.pyplot as plt

def read_data_from_csv(data_path : str) -> dict:
    """
    This function reads the csv file and create a dictionary with
    interest description as the key and name as the value

    Args:
        data_path (string): path for the csv file

    Returns:
        dictionary: interest description as the key and name as the value
    """
    attendees_map = {}
    with open(data_path, newline="", encoding='utf-8') as csvfile:
        attendees = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(attendees)  # Skip the header row
        for row in attendees:
            name, paragraph = row
            attendees_map[paragraph] = name

    return attendees_map


def create_embeddings(model, attendees_map):
    paragraphs = list(attendees_map.keys())
    embeddings = model.encode(paragraphs)
        
    # Create a dictionary to store embeddings for each person
    person_embeddings = {attendees_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}
    
    return person_embeddings

def scale_data(embeddings):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(list(embeddings.values()))
    return scaled_data
    
    
def dimensionality_reduction(embeddings):
    reducer = umap.UMAP(random_state=C.RANDOM_STATE)
    reduced_data = reducer.fit_transform(embeddings)
    return reduced_data

def plot_and_save_visualization(embeddings, reduced_embeddings, plot_path = 'visualization.png'):
    # Creating lists of coordinates with accompanying labels
    x = [row[0] for row in reduced_embeddings]
    y = [row[1] for row in reduced_embeddings]
    label = list(embeddings.keys())

    # Plotting and annotating data points   
    plt.scatter(x,y)
    for i, name in enumerate(label):
        plt.annotate(name, (x[i], y[i]), fontsize="3")

    # Clean-up and Export
    plt.axis('off')
    plt.savefig(os.path.join(C.BASE_RESULTS_PATH , plot_path), dpi=800)
    
    return None



# Create a function to be called while serializing JSON
def json_serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def compare_embedding(str1: str, str2: str) -> float:
    """
    Given two strings, compare their embeddings' cosine similarity.
    """
    model = SentenceTransformer(m.MINILM_L6_V2)
    embedding_1 = model.encode(str1)
    embedding_2 = model.encode(str2)

    cosine_similarity = 1 - cosine(embedding_1, embedding_2)
    return cosine_similarity