import operator
from collections import Counter
from wktools.wkutils import get_tokens


class WikiGraph:
    """
    Wikipedia graph representation

    Parameters
    ----------
    pages : list of dict
        List of Wikipedia page (a page contain a 'title' and 'content' which is the source code)
    n : int
        Number of common token needed to create a relation between two pages

    """
    def __init__(self, pages, n=1):
        self.size = len(pages)
        self.pages = pages
        self.list_tokens = []  # will be used to find cluster's title
        self.adj_matrix = [[0 for _2 in range(self.size)] for _ in range(self.size)]

        tokens_list = []  # store extracted tokens for each page
        # extract all tokens from pages
        for i, page in enumerate(pages):
            tokens = get_tokens(page['content'])
            tokens_set = set(tokens)
            # update adjacency matrix
            for k, compared_tokens in enumerate(tokens_list):
                if len(tokens_set.intersection(compared_tokens)) >= n:
                    self.adj_matrix[i][k] = 1
                    self.adj_matrix[k][i] = 1
            self.list_tokens.append(tokens)
            tokens_list.append(tokens_set)

    def __get_connected_component(self, node, visited_nodes):
        """
        Retrieve nodes connected to input node

        Parameters
        ----------
        node : int
            Starting node
        visited_nodes : list of int
            Already visited nodes (it will be updated by the function)

        Returns
        -------
        connected_comp : set of int
            Connected component
        """
        connected_comp = {node}
        for node2, adj in enumerate(self.adj_matrix[node]):
            if adj == 1 and not visited_nodes[node2]:
                visited_nodes[node2] = True
                connected_comp.add(node2)
                new_comp = self.__get_connected_component(node2, visited_nodes)
                connected_comp = connected_comp.union(new_comp)

        return connected_comp

    def connected_components(self):
        """
        Get all connected components of the graph (Deep First Search algorithm)

        Returns
        -------
        connected_comps : list of set of int
            List of connected components
        """
        connected_comps = []
        visited_nodes = [False] * self.size
        for node in range(self.size):
            if not visited_nodes[node]:
                connected_comp = self.__get_connected_component(node, visited_nodes)
                connected_comps.append(connected_comp)
        return connected_comps

    def clusters(self):
        """
        Get clusters (connected components associated to a title)

        Returns
        -------
        clusters : list of dict
            Clusters (composed of a 'title' and 'connected_comp' a set of the component's node's title)
        """
        # used to calculate the total frequency of tokens
        total_tokens = [item for sublist in self.list_tokens for item in sublist]
        count_total_tokens = Counter(total_tokens)
        n_total = len(total_tokens)

        connected_comps = self.connected_components()
        clusters = []
        for connected_comp in connected_comps:
            # group all token in the same component
            tokens = []
            for comp in connected_comp:
                tokens += self.list_tokens[comp]

            # used to calculate the frequency of tokens in a connected component
            count_tokens = Counter(tokens)
            n = len(tokens)

            freq_tokens = {}
            for i, token in enumerate(count_tokens.keys()):
                # calculate a variant of TF-IDF in order to get word's importance
                freq_tokens[token] = (count_tokens[token]/n) / (count_total_tokens[token]/n_total)

            title = max(freq_tokens.items(), key=operator.itemgetter(1))[0]  # get most important word
            connected_nodes_title = [self.pages[node]['title'] for node in connected_comp]  # use article's title
            clusters.append({'title': title, 'connected_comp': connected_nodes_title})
        return clusters
