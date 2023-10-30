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
        
    def get_mapped_data(self):
        self.mapped_data = U.read_data_from_csv(self.data_src)

    def modelling(self):
        self.model = SentenceTransformer(self.model_src)
        
    def create_embeddings(self):
        self.mapped_embeddings = U.create_embeddings(self.model, self.mapped_data)
        
    def preprocessing(self):
        self.embeddings = U.scale_data(self.mapped_embeddings)

    def dimensionality_reduction(self):
        self.dim_reduced_embeddings = U.dimensionality_reduction(self.embeddings)
    
    def create_visualization(self):
        U.plot_and_save_visualization(self.mapped_embeddings, self.dim_reduced_embeddings)
        

if __name__ == "__main__":
    MM = MatchMaker(C.DATA_PATH, C.MODEL_PATH)
    MM.get_mapped_data()
    MM.modelling()
    MM.create_embeddings()
    MM.preprocessing()
    MM.dimensionality_reduction()
    MM.create_visualization()
    # print(MM.dim_reduced_embeddings)
    
    
