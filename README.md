# Matchmaking with Embedding Vectors

_"Words can't describe how unique your interests are... but coordinates can" - Sean Ashley, circa 2023_

A flattened embedding space of names clustered based on their interests using the sentence-transformers all-MiniLM-L6-v2 model. Created for the UW Startups S23 Kickoff event with guidance from [Jacky Zhao](https://jzhao.xyz/) and [Sean Ashley](https://www.linkedin.com/in/sean-ashley).

![Sample output of script](https://github.com/raoofnaushad/matcher/blob/main/visualization.png?raw=true)

## Instructions for use

1. Collect or format your data in the following format

| Name  | What are your interests? (or varying permutations of this question) |
| ----- | ------------------------------------------------------------------- |
| Alice | I love being the universal placeholder for every CS joke ever       |
| Bob   | I too love being the universal placeholder for every CS joke        |

2. Clone the repository
3. Replace `attendees.csv` in `visualizer.ipynb` with the path to your downloaded data
4. Run all cells
5. Bask in the glory of having an awesome new poster


## Step - 1: Setting up the environment

- Prerequisite: Install anaconda or miniconda (Or you can create the environment using python virtualenv use the packages mentioned in the environment.yml file)
- 
