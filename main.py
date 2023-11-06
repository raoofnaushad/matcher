
import os

import config as C
import utils as U

from sentence_transformers import SentenceTransformer



class MatchMaker:
    """_summary_
    """
    def __init__(self, 
                _type = 'generate',
                file_name = '',
                model = '',
                embeddings_path = '',
                dim_reduction_method = 'UMAP',
                plot_path = 'visualization.png'):
        
        self._type = _type
        self.data_src = os.path.join(C.BASE_DATA_PATH, file_name)
        self.model_src = model
        self.embeddings_path = os.path.join(C.BASE_RESULTS_PATH, embeddings_path)
        self.dim_reduction_method = dim_reduction_method
        self.plot_path = os.path.join(C.BASE_RESULTS_PATH, plot_path)
        
        self.model = SentenceTransformer(self.model_src)
        self.mapped_data = U.read_data_from_csv(self.data_src)
    
    
    def run(self):
        print(f"Started creating embeddings from data: {self.data_src}")
        self.mapped_embeddings, self.embeddings = U.create_embeddings(self.model, self.mapped_data, self.embeddings_path, scale_data=True, save_embeddings=True)
        print(f"Embeddings created successfully: {self.embeddings_path}")
        print("-----"*10)
        
        print(f"Dimensionality reduction using {self.dim_reduction_method}")
        self.dim_reduced_embeddings = U.dimensionality_reduction(self.mapped_embeddings, self.embeddings, self.dim_reduction_method, self.embeddings_path, save_embeddings=True)
        print(f"Dimensionality reduction completed successfully using {self.dim_reduction_method}")
        print("-----"*10)
        
        print(f"Creating plot for the dimensionality reduced data")
        U.plot_and_save_visualization(self.mapped_embeddings, self.dim_reduced_embeddings, self.plot_path)  
        print(f"Saved plot in: {self.plot_path}")
        print("-----"*10)
        

        if self._type == 'generate':
            pass

if __name__ == "__main__":
    types = {
        "generate",
        "data-analysis",
        "model-comparison",
        
    }
    
    MM = MatchMaker(_type = 'generate', 
                    file_name = 'classmates.csv', 
                    model = C.MINILM_L6_V2, 
                    embeddings_path = 'person_embeddings.json',
                    dim_reduction_method = 'UMAP',
                    plot_path = 'person_embeddings.png')
    
    MM.run()
    
    
