from src import config as C  ## User defined file for all constants
from src import utils as U  ## User defined file for all constants

from sentence_transformers import SentenceTransformer


class MatchMaker:
    """_summary_
    """
    def __init__(self, data_src, model_src):
        self.data_src = data_src
        self.model_src = model_src
        self.mapped_data = dict()
        # self.model = SentenceTransformer(self.model_src)
        
        
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
        
        
    def dimensionality_reduction(self):
        ## dimensionality reduction
        self.dim_reduced_embeddings = U.dimensionality_reduction(self.embeddings)
        
        
    def visualization(self):
        ## Plotting the plot
        ## Saving the plot
        U.plot_and_save_visualization(self.mapped_embeddings, self.dim_reduced_embeddings)    

        

if __name__ == "__main__":
    MM = MatchMaker(C.DATA_PATH, C.MODEL_PATH)
    MM.data()
    MM.embedding()
    MM.dimensionality_reduction()
    MM.visualization()

    
    
