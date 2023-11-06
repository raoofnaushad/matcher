
import os

import config as C
import utils as U
from src import data_comparison

from sentence_transformers import SentenceTransformer, util


class MatchMaker:
    """_summary_
    """
    def __init__(self, 
                _type = 'generate',
                file_name = '',
                model = '',
                embeddings_path = '',
                dim_reduction_method = 'UMAP',
                dim_components = 3,
                plot_title = "Embeddings Plot",
                plot_path = 'visualization.png'):
        
        self._type = _type
        self.data_src = os.path.join(C.BASE_DATA_PATH, file_name)
        self.model_src = model
        self.embeddings_path = os.path.join(C.BASE_RESULTS_PATH, embeddings_path)
        self.dim_reduction_method = dim_reduction_method
        self.plot_title = plot_title
        self.plot_path = os.path.join(C.BASE_RESULTS_PATH, plot_path)
        
        self.model = SentenceTransformer(self.model_src)
        self.mapped_data = U.read_data_from_csv(self.data_src)
        self.dim_components = dim_components
    
    
    def run(self):
        print(f"Started creating embeddings from data: {self.data_src}")
        self.mapped_embeddings, self.embeddings = U.create_embeddings(self.model, self.mapped_data, self.embeddings_path, scale_data=True, save_embeddings=True)
        print(f"Embeddings created successfully: {self.embeddings_path}")
        print(f"Embeddings Dimension is: {len(self.embeddings[0])}")
        print("-----"*10)

        
        print(f"Dimensionality reduction using {self.dim_reduction_method}")
        self.dim_reduced_embeddings = U.dimensionality_reduction(self.mapped_embeddings, self.embeddings, self.dim_reduction_method, self.embeddings_path, self.dim_components, save_embeddings=True)
        print(f"Dimensionality reduction completed successfully using {self.dim_reduction_method}")
        print(f"Reduced Embeddings Dimension is: {len(self.dim_reduced_embeddings[0])}")
        print("-----"*10)
        
        print(f"Creating plot for the dimensionality reduced data")
        U.plot_and_save_visualization(self.mapped_embeddings, self.dim_reduced_embeddings, self.plot_title, self.plot_path)  
        print(f"Saved plot in: {self.plot_path}")
        print("-----"*10)

        if self._type == 'generate':
            pass
        if self._type == 'data-analysis':
            print(f"Analysing Cosine Similarity for the modified sentences")
            similarity_metric = data_comparison.compare_modified_vector_embeddings()

            print(f"Please see the below cosine similarity analysis")
            for each in similarity_metric:
                print(f"Similarity metric of the two sentences of {each} is: {similarity_metric[each]}")
            print("-----"*10)
        

if __name__ == "__main__":
    types = {
        "generate",
        "data-analysis",
        "model-comparison",
        
    }
    
    # ## 1. Generate and Visualize ClassMates Data
    # MM = MatchMaker(_type = 'generate', 
    #                 file_name = 'classmates.csv', 
    #                 model = C.MINILM_L6_V2, 
    #                 embeddings_path = 'person_embeddings.json',
    #                 dim_reduction_method = 'UMAP',
    #                 dim_components = 2,
    #                 plot_title = "MCDA Classmates Embeddings",
    #                 plot_path = 'person_embeddings.png')
    
    
    
    ## 2. Data Analysis - Checking the impact of sentence changes!
    MM = MatchMaker(_type = 'data-analysis', 
                    file_name = 'classmates_analysis.csv', 
                    model = C.MINILM_L6_V2, 
                    embeddings_path = 'data_analysis_embeddings.json',
                    dim_reduction_method = 'UMAP',
                    dim_components = 2,
                    plot_title = "Embeddings Analysis",
                    plot_path = 'data_analysis.png')
    
    
    
    MM.run()
    
    
