from tests.test_wkclustering import TestWKClusteringMethods
from wktools.wkclassification import WikiClassifier, WikiDataTransformer
from wktools.wkclustering import WikiGraph
from wktools.wkpipeline import DataLoader
from wktools.wkutils import get_tokens

data_filename = "data/dataset_business_technology_cybersecurity.pickle"
data = DataLoader()(data_filename)
contents = [page['content'] for page in data]
topics = [page['topic'] for page in data]

"""
############################################

          ######## PART 1 ########
          
############################################

The code for this part can be found in wktools/wkclustering.py

"""

"""
### A.1 ###
"""
graph = WikiGraph(data, 400)

"""
### A.2 ###
"""
# The complexity is O(M * N**2) where M is the average length of a web page (M come from the get_tokens() function)
# and N is the number of webpages (N**2 come from the nested loops).

"""
### A.3 ###
"""
print(graph.connected_components())

"""
### A.4 ###
"""
# The complexity is O(|E| + |N|) where |E| is the number of edges and |N| is the number of nodes

"""
### A.5 ###
"""
# We can select the word with the highest TF-IDF (term frequency * inverse document frequency) (or a variant of it)
# that show the importance of a word in a document (or the entire connected component in our case).

"""
### A.6 ###
"""
# Here we get a big cluster named 'financials' with 141 nodes, it seems to be a good cluster considering our dataset.
print(graph.clusters())

"""
### B.1 ###
"""
# See main.py

"""
### B.2 ###
"""
# run with "python main.py -f 'data/dataset_business_technology_cybersecurity.pickle'"

"""
### B.3 ###
"""
# Honestly I think the quality is not that good, even after trying to finely tune the parameter n (in the graph
# creation), I feel like I either get a really big cluster or some isolated nodes (although we expect something like
# 3 big clusters since we have 3 categories of articles). To solve the problem we could do a list of really frequent
# word used in wikipedia and remove them from our tokenized page so that there are less common tokens between
# articles, but the one that are common are more relevant.

"""
### B.4 ###
"""
# I think we could use latent semantic approach with algorithms like LDA/PLSA in order to find the different topics.

"""
### B.5 ###
"""
# It is better to run 'python -m unittest tests/test_wkclustering.py -v'
tests = TestWKClusteringMethods()
tests.test_clustering()
tests.test_graphs()
tests.test_tokens()

"""
############################################

          ######## PART 2 ########

############################################

The code for this part can be found in wktools/wkclassification.py

"""

"""
### 1 ###
"""
# We first get our vocabulary
tokens = set()
for x in contents:
    tokens = tokens.union(set(get_tokens(x)))

data_transformer = WikiDataTransformer(tokens, set(topics))
train_dataset, test_dataset, val_dataset = data_transformer.transform_split(contents, topics, 0.7, 0.15)
model = WikiClassifier(data_transformer, 1000)
model.build((data_transformer.max_size, 10))
print(model.summary())
model.compile()

"""
### 2 ###
"""
# I think that the best metric is the recall. It is important to be sure to suggest every articles that could
# interest the reader even if we suggest accidentally some articles that do not concern him. If we miss an article
# that would have interested the reader it do more damage (in this case the reader will miss the information) than
# if we suggest an article not interesting (in which case the reader can simply not open it).


"""
### 3 ###
"""
model.fit(train_dataset, epochs=20, validation_data=val_dataset)
scores = model.evaluate(test_dataset)
print("Recall: %.2f%%" % (scores[1]*100))


"""
### 4 ###
"""
# The metrics are important (as we saw earlier for the recall) but we also need to consider the resources used in
# order to run the model (in time, memory and cpu/gpu) and if it is scalable.

"""
### Conclusion ###
"""
# I am  really satisfied for what I accomplished with the time I had. It is possible that my code could be more
# optimized (be faster and with more sanity checks) but I wanted to keep a code clear and I think that I
# accomplished that. As I said in the previous answer, clustering isn't perfect and I have some leads to improve it.
# Overall I found this subject really interesting (and it was really clear) and I'm eager to receive your feedback.
