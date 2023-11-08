# Matchmaking with Embedding Vectors

_"Words can't describe how unique your interests are... but coordinates can" - Sean Ashley, circa 2023_

A flattened embedding space of names clustered based on their interests using the sentence-transformers all-MiniLM-L6-v2 model. Created for the UW Startups S23 Kickoff event with guidance from [Jacky Zhao](https://jzhao.xyz/) and [Sean Ashley](https://www.linkedin.com/in/sean-ashley).

![Sample output of script](https://github.com/raoofnaushad/matcher/blob/main/results/person_embeddings.png?raw=true)

## Packages Used

### 1. Model for Embeeding Vector Generation
The Model package used in this project is obtained from the Python package called [SentenceTransformers](https://www.sbert.net/). SentenceTransformers is a Python framework for state-of-the-art sentence, text, and image embeddings. 

#### Model Options
- **Model Name**: The model used in this project is `all-mpnet-base-v2`, which provides the best quality embeddings.
- **Alternative Option**: If you require faster processing, you can opt for the `all-MiniLM-L6-v2` model, which is approximately 5 times faster while still offering good quality sentence embeddings.

#### About SentenceTransformers
SentenceTransformers is a Python framework that implements state-of-the-art techniques for generating embeddings for sentences, text, and images. The initial work on this framework is described in the paper "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."


## What are Embeddings? 

Embeddings act as a form of translation for computers, equipping them with the ability to decipher the underlying meaning of language. In this process, each word is assigned a unique numerical vector, essentially serving as a distinct identifier for that word. These vectors are meticulously designed to ensure that words with similar meanings have vectors with similarities. Consequently, words like "apple" and "fruit" would be situated closer in this vector space, denoting their related meanings.

### Sentence Embeddings

Furthermore, the concept of embeddings extends to entire sentences. Sentence embeddings encapsulate the essence of a sentence within a numerical vector. This allows a computer to comprehend the overall meaning of a sentence, even within complex contexts. The capacity to represent words and sentences as numerical vectors empowers computers to execute various language-related tasks, ranging from translation to sentiment analysis. These vectors can be effectively compared and manipulated, enabling computers to make sense of language in a highly efficient manner.

### Visualizing Embeddings

As illustrated in the image above, which is generated from the classmates.csv file containing people's names and their hobbies, visualizing embeddings in a 2D space unveils a compelling pattern. Individuals with similar interests or hobbies are clustered closely together. It's akin to crafting a map of language that allows computers to navigate and understand the relationships between words and sentences. These embeddings act as a bridge that translates language into a code that computers can comprehend, making them invaluable tools in the realm of natural language processing.


## Data Analysis

In our data analysis, we aimed to investigate the impact of word changes on sentence embeddings generated by Sentence Transformers. We made both minor and major modifications to the data to explore how these alterations affected the resulting embedding vectors and their logical representations.

### Data Modifications
We conducted several experiments by crafting modified sentences for five classmates. Notably, our observations were derived from both the plotted generated embeddings and the cosine similarity metric.

![Modified Sentences](https://github.com/raoofnaushad/matcher/blob/main/results/modified_sentences.png?raw=true)

<br>

![Similarity Match After Modifying Sentences](https://github.com/raoofnaushad/matcher/blob/main/results/data_analysis.png?raw=true)

<br>

![Cosine Similarity Metric](https://github.com/raoofnaushad/matcher/blob/main/results/similarity_metric.png?raw=true)


- **Synonymous Changes:** <br>
For Umair, Deepak, and Neeyati, we aimed to maintain the same sentence meaning while modifying individual words with synonyms or similar words. When we compared their original sentences with these modified versions, the cosine similarity was remarkably high. Umair's sentence with a single word change exhibited a 97.5% cosine similarity, indicating that the embeddings captured the synonymous meaning effectively. Similarly, Neeyati and Deepak's sentences, although composed of different words, maintained a significant plot proximity with cosine similarity scores exceeding 70%. This suggests that Sentence Transformers can successfully capture the semantic nuances of words and their impact on the overall meaning of a sentence.

- **Semantic Inversion:** <br>
Conversely, for Greg, we deliberately changed the sentence to convey the opposite meaning. While initially, Greg was described as a person who loves the outdoors, we transformed the sentence to depict a preference for indoor activities. The embeddings of these opposing sentences were notably distant in the plot, and the cosine similarity scores confirmed their dissimilarity. This underscores the ability of Sentence Transformers to discern significant shifts in sentence meaning due to word changes.

- **Challenges in Contextual Understanding:** <br>
Lastly, in the case of Samuel, we retained the same words but reversed the intended meaning from someone who loves activities like movies and badminton to someone who despises them. Surprisingly, the embeddings for these contrasting sentences still exhibited a relatively high cosine similarity of 82.5%. This suggests that while Sentence Transformers excel in understanding individual word meanings, they can sometimes struggle to preserve the context of a whole sentence when multiple common words are present.

**In summary**, our data analysis reveals that Sentence Transformers are highly effective in capturing the meaning of individual words, even when they are replaced with synonyms or antonyms. However, the context of a sentence can be influenced by the presence of shared words, leading to limitations in preserving the overall sentence meaning. This understanding is crucial for utilizing sentence embeddings effectively in natural language processing tasks.


## Embedding Sensitivity Tests:

Our analysis delves into the sensitivity of our results to the choice of model for generating embeddings. Specifically, we compared the embeddings created using two Sentence Transformers models: 1. all-MiniLM-L6-v2 and 2. all-mpnet-base-v2. By taking one individual (e.g., Greg Kirczenow), we assessed the cosine difference and rank difference between Greg and all other classmates when using these two distinct models. This examination provides insights into the sensitivity of embeddings generated from different models.

![Cosine Diff and Rank Correlation](https://github.com/raoofnaushad/matcher/blob/main/results/model_comparison.png?raw=true)

- **Quantitative Considerations:** <br>
We observed notable differences in rank correlations between the reference person (Greg) and other classmates when using the two models. Rank correlation is determined by calculating the cosine similarity between the reference person's embedding and the embeddings of all other individuals. The distinct ranks signify that the embeddings generated by the two models are sensitive to model choice.

- **Qualitative Considerations:** <br>
To exemplify this sensitivity, let's take an illustrative case. With the first model (all-MiniLM-L6-v2), the closest classmates to Greg are Sylvester Terdoo, D'Shon Henry, and Royston Furtado. However, when employing the second model (all-mpnet-base-v2), the closest classmates to Greg change to D'Shon Henry, Nikita Neveditsin, and Pawan Lingras. Remarkably, there is only one common individual among the top three closest classmates in the two models. This highlights that different embeddings generated from different models can significantly alter the relative proximity of individuals, indicating the model's influence on the results.

**In summary**, our sensitivity tests reveal that the choice of model for generating embeddings plays a crucial role in the outcomes. The differences in rank correlations and the reordering of closest classmates underscore the sensitivity of embeddings to the specific model used, emphasizing the need for careful consideration when selecting a model for a given natural language processing task.



## Effect of RandomSeed on UMAP:

![Image 1](https://github.com/raoofnaushad/matcher/blob/main/results/withoutseed1.png?raw=true)

<br>

![Image 2](https://github.com/raoofnaushad/matcher/blob/main/results/withoutseed2.png?raw=true)

When changing the seed for the UMAP function and rerunning the code, you'll notice variations in the generated visualization (visualization.png). These differences stem from the random weight initialization in the UMAP algorithm. However, despite the visual discrepancies, the relative relationships between vectors remain consistent. For example, individuals like Akash, Mehul, and Francis maintain their proximity in both visualizations. Setting a seed is crucial for reproducibility, ensuring that you obtain consistent results when running the UMAP algorithm multiple times.
