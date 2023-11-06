
from src import config as C  ## User defined file for all constants
from src import utils as U  ## User defined file for all constants

from sentence_transformers import SentenceTransformer

import os
import json


class MatchMaker:
    """_summary_
    """
    def __init__(self, data_src, model_src, plot_path):
        self.data_src = data_src
        self.model_src = model_src
        self.mapped_data = dict()
        # self.model = SentenceTransformer(self.model_src)
        self.plot_path = plot_path
        
        
    def data(self):
        """
            This function will deal with reading data from source, preprocessing etc.
        """
        ## Reading Data
        self.mapped_data = U.read_data_from_csv(self.data_src)
        ## Preprocessing Data
        ## So and So
        
    def embedding(self):
        ## Model
        self.model = SentenceTransformer(self.model_src)
        ## creating embeddings
        self.mapped_embeddings = U.create_embeddings(self.model, self.mapped_data)
        ## Scaling embeddings
        self.embeddings = U.scale_data(self.mapped_embeddings)    
        ## Save embeddings
        with open(C.MOD_PERSON_EMBEDDINGS_PATH, 'w') as f:
            json.dump(self.mapped_embeddings, f, default=U.json_serialize)
        
        
    def dimensionality_reduction(self):
        ## dimensionality reduction
        self.dim_reduced_embeddings = U.dimensionality_reduction(self.embeddings)
        

    # def compare_embeddings(self, keyWord='Modified'):
    #     ## Comparing the embeddings of modified people
        
        
        
    def visualization(self):
        ## Plotting the plot
        ## Saving the plot
        U.plot_and_save_visualization(self.mapped_embeddings, self.dim_reduced_embeddings, self.plot_path)
      

        

if __name__ == "__main__":
    MM = MatchMaker(C.CLASSMATES_MOD_DATA_PATH, C.MODEL_PATH, C.MOD_PLOT_PATH)
    MM.data()
    MM.embedding()
    MM.dimensionality_reduction()
    MM.visualization()

    
    
