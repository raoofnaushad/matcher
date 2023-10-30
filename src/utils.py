from src import config as C

import csv
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

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

def plot_and_save_visualization(embeddings, reduced_embeddings):
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
    plt.savefig('visualization.png', dpi=800)
    
    return None