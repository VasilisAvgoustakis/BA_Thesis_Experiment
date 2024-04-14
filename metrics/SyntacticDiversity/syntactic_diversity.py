import spacy
from grakel import GraphKernel, Graph
import numpy as np


# Load a spaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")

def construct_dependency_graphs(sentences):
    graphs = []
    for sentence in sentences:
        doc = nlp(sentence)
        adjacency_list = {}
        node_labels = {}  # A dictionary to store the labels of nodes
        
        for token in doc:
            # We'll use token index as the node identifier
            adjacency_list[token.i] = [child.i for child in token.children]
            node_labels[token.i] = token.lemma_  # Using lemma as a simple node label

        # Create a Grakel Graph with node labels
        g = Graph(adjacency_list, node_labels=node_labels, graph_format='adjacency')
        graphs.append(g)
    return graphs



def calculate_syntactic_diversity(graphs, measure='wl'):
    kernel = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 5}, {"name": "subtree_wl"}], normalize=True)
    # Compute the kernel matrix
    K = kernel.fit_transform(graphs)
    
    if measure == 'wl':  # Weisfeiler-Lehman kernel measure
        # Use kernel matrix to calculate diversity; more similar structures will have higher kernel values
        diversity_scores = 1 - np.mean(K, axis=1)  # Diversity as 1 - average similarity
        syntactic_diversity = np.mean(diversity_scores)
    else:
        raise ValueError("Unsupported measure for syntactic diversity")
    return syntactic_diversity