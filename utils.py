
import csv
import json

import numpy as np
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import matplotlib.pyplot as plt

import config as C



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


def json_serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def create_embeddings(model, attendees_map, embeddings_path, scale_data=True, save_embeddings=True):

    paragraphs = list(attendees_map.keys())
    embeddings = model.encode(paragraphs)
        
    # Create a dictionary to store embeddings for each person
    person_embeddings = {attendees_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}
    
    if scale_data:
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(list(person_embeddings.values()))    
        
            
    if save_embeddings:
        with open(embeddings_path, 'w') as f:
            json.dump(person_embeddings, f, default=json_serialize)    
    
    return person_embeddings, scaled_embeddings


def dimensionality_reduction(mapped_embeddings, embeddings, dim_red_method, embeddings_path, save_embeddings=True):
    if dim_red_method == 'UMAP':
        reducer = umap.UMAP(random_state=C.RANDOM_STATE)
        reduced_data = reducer.fit_transform(embeddings)

    pupils = list(mapped_embeddings.keys())
    # Create a dictionary to store embeddings for each person
    person_red_embeddings = {user: embedding for user, embedding in zip(pupils, reduced_data)}
    

    if save_embeddings:
        with open(embeddings_path.split('.')[0] + '_dim_red.json', 'w') as f:
            json.dump(person_red_embeddings, f, default=json_serialize)    
    
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
    # plt.show()
    plt.savefig(plot_path, dpi=800)
    