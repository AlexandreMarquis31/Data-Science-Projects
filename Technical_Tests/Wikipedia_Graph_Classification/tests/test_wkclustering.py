import unittest
from wktools import wkutils
from wktools.wkclustering import WikiGraph
from random import randint, choice
import string

# dummy data (keywords which will help generating dummy wiki pages)
dummy_pages_list = [
    {'title': 'Ice-cream flavours', 'tokens': ['chocolate', 'caramel', 'vanilla', 'mango', 'ice-cream']},
    {'title': 'Top 10 desserts', 'tokens': ['chocolate', 'cake', 'cacao', 'dark']},
    {'title': 'Autonomous cars', 'tokens': ['robot', 'car', 'future', 'autonomous']},
    {'title': "Kid's favorite foods", 'tokens': ['lollipop', 'cake', 'ice-cream']},
    {'title': 'Transports', 'tokens': ['plane', 'car', 'train', 'subway', 'future']},
    {'title': 'World cup 2018', 'tokens': ['football', 'field', 'ball', 'player', 'coach', 'soccer', 'tournament']},
    {'title': 'Sports', 'tokens': ['ball', 'coach', 'valley', 'player', 'tournament']},
    {'title': 'Cars', 'tokens': ['car', 'drive', 'toyota', 'oil']},
    {'title': 'Vanilla', 'tokens': ['vanilla', 'flower', 'madagascar', 'dessert', 'cake']},
    {'title': 'Countries', 'tokens': ['madagascar', 'france', 'russia', 'australia', 'spain']}
]

dummy_tokens1 = ['chocolate', 'caramel', 'vanilla', 'mango', 'ice-cream']
dummy_tokens2 = ['chocolate', 'cake', 'cacao', 'dark']
dummy_tokens3 = ['robot', 'car', 'future', 'autonomous']
dummy_tokens4 = ['lollipop', 'cake', 'ice-cream']
dummy_tokens5 = ['plane', 'car', 'train', 'subway', 'future']
dummy_tokens6 = ['football', 'field', 'ball', 'player', 'coach', 'soccer', 'tournament']
dummy_tokens7 = ['ball', 'coach', 'valley', 'player', 'tournament']
dummy_tokens8 = ['car', 'drive', 'toyota', 'oil']
dummy_tokens9 = ['vanilla', 'flower', 'madagascar', 'dessert', 'cake']
dummy_tokens10 = ['madagascar', 'france', 'russia', 'australia', 'spain']

# hand made computed adjacency matrices for graphs when n = 1
ajd_matrix_1 = [
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
]

# hand made computed adjacency matrices for graphs when n = 2
ajd_matrix_2 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]


class TestWKClusteringMethods(unittest.TestCase):
    """
    Unit-tests for wkclustering.
    """
    def test_tokens(self):
        """
        Test tokenization of Wikipedia's pages
        """
        dummy_contents = self.__generate_dummy_contents__(dummy_pages_list)
        wkgraph = WikiGraph(dummy_contents)
        for i, tokens in enumerate(wkgraph.list_tokens):
            self.assertEqual(set(dummy_pages_list[i]['tokens']), set(tokens))

    def test_graphs(self):
        """
        Test graph creation, we check the created adjacency matrix
        """
        dummy_contents = self.__generate_dummy_contents__(dummy_pages_list)
        wkgraph1 = WikiGraph(dummy_contents)
        self.assertEqual(wkgraph1.adj_matrix, ajd_matrix_1)

        wkgraph2 = WikiGraph(dummy_contents, 2)
        self.assertEqual(wkgraph2.adj_matrix, ajd_matrix_2)

        wkgraph10 = WikiGraph(dummy_contents, 10)
        # there should be no adjacency
        for adj_node in wkgraph10.adj_matrix:
            for adj in adj_node:
                self.assertEqual(adj, 0)

    def test_clustering(self):
        """
        Test graph clustering
        """
        dummy_contents = self.__generate_dummy_contents__(dummy_pages_list)
        wkgraph1 = WikiGraph(dummy_contents)
        self.assertEqual(wkgraph1.clusters()[0]['title'], 'ice-cream')
        self.assertEqual(wkgraph1.clusters()[0]['connected_comp'], ['Ice-cream flavours', 'Top 10 desserts', "Kid's favorite foods", 'Vanilla', 'Countries'])
        self.assertEqual(wkgraph1.clusters()[1]['title'], 'autonomous')
        self.assertEqual(wkgraph1.clusters()[1]['connected_comp'], ['Autonomous cars', 'Transports', 'Cars'])

        wkgraph2 = WikiGraph(dummy_contents, 2)
        # check that every node is in a cluster
        self.assertEqual(set().union(*[cluster['connected_comp'] for cluster in wkgraph2.clusters()]), set([page['title'] for page in dummy_pages_list]))

        self.assertEqual(wkgraph2.clusters()[2]['title'], 'autonomous')
        self.assertEqual(wkgraph2.clusters()[2]['connected_comp'], ['Autonomous cars', 'Transports'])
        self.assertEqual(wkgraph2.clusters()[4]['title'], 'tournament')
        self.assertEqual(wkgraph2.clusters()[4]['connected_comp'], ['World cup 2018', 'Sports'])
        self.assertEqual(len(wkgraph2.clusters()), 8)

        wkgraph10 = WikiGraph(dummy_contents, 10)
        # there should be no adjacency so each node is a cluster
        self.assertEqual(len(wkgraph10.clusters()), 10)

    @staticmethod
    def __generate_dummy_contents__(pages_list):
        """
        Generate our dummy source codes from the "dummy_tokens"
        Parameters
        ----------
        pages_list : list of dict
            List of keywords and titles used to generate the page

        Returns
        -------
        dummy_contents : list of str
            List of dummy wiki page source code
        """
        dummy_contents = []
        for page in pages_list:
            dummy_content = TestWKClusteringMethods.__generate_dummy_content__(page)
            dummy_contents.append(dummy_content)
        return dummy_contents

    @staticmethod
    def __generate_dummy_content__(page):
        """
        Generate a dummy source code from some keywords
        Parameters
        ----------
        page : dict
            Keywords and title used to generate the page

        Returns
        -------
        content : str
            Dummy wiki source code
        """
        letters = string.ascii_lowercase
        punctuation = [punct for punct in string.punctuation if punct not in ['<', '>']]
        content = ""
        tmp_tokens = page['tokens'].copy()
        while len(tmp_tokens) != 0:
            random_i = randint(0, 10)
            for i in range(random_i):  # add some random stopwords and a punctuation
                content += ' ' + choice(list(wkutils.stopwords)) + ' ' + choice(punctuation) + ' '
            if randint(0, 1) == 1:   # add a pseudo html tag
                tag = ''.join([choice(letters) for i in range(6)])
                content += " <" + tag + "> " + tmp_tokens.pop() + " </" + tag + "> "
            else:
                content += ' ' + tmp_tokens.pop().upper() + ' '
        new_page = {'title' : page['title'], 'content': content }
        return new_page


if __name__ == '__main__':
    unittest.main()
