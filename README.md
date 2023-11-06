# Matchmaking with Embedding Vectors

_"Words can't describe how unique your interests are... but coordinates can" - Sean Ashley, circa 2023_

A flattened embedding space of names clustered based on their interests using the sentence-transformers all-MiniLM-L6-v2 model. Created for the UW Startups S23 Kickoff event with guidance from [Jacky Zhao](https://jzhao.xyz/) and [Sean Ashley](https://www.linkedin.com/in/sean-ashley).

![Sample output of script](https://github.com/raoofnaushad/matcher/blob/main/visualization.png?raw=true)


## What are Embeddings? 

Embeddings act as a form of translation for computers, equipping them with the ability to decipher the underlying meaning of language. In this process, each word is assigned a unique numerical vector, essentially serving as a distinct identifier for that word. These vectors are meticulously designed to ensure that words with similar meanings have vectors with similarities. Consequently, words like "apple" and "fruit" would be situated closer in this vector space, denoting their related meanings.

### Sentence Embeddings

Furthermore, the concept of embeddings extends to entire sentences. Sentence embeddings encapsulate the essence of a sentence within a numerical vector. This allows a computer to comprehend the overall meaning of a sentence, even within complex contexts. The capacity to represent words and sentences as numerical vectors empowers computers to execute various language-related tasks, ranging from translation to sentiment analysis. These vectors can be effectively compared and manipulated, enabling computers to make sense of language in a highly efficient manner.

### Visualizing Embeddings

As illustrated in the image above, which is generated from the classmates.csv file containing people's names and their hobbies, visualizing embeddings in a 2D space unveils a compelling pattern. Individuals with similar interests or hobbies are clustered closely together. It's akin to crafting a map of language that allows computers to navigate and understand the relationships between words and sentences. These embeddings act as a bridge that translates language into a code that computers can comprehend, making them invaluable tools in the realm of natural language processing.