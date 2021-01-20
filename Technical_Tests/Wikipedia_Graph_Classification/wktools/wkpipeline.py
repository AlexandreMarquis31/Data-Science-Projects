import io
import pickle
from wktools.wkclustering import WikiGraph


class Pipeline:
    """
    Standard pipeline

    Parameters
    ----------
    steps : list
        List of actions to be executed by the pipeline (use a tuple for actions with args)
    """
    def __init__(self, steps):
        self.actions = []
        self.args = []
        for step in steps:
            if isinstance(step, tuple):
                self.actions.append(step[0])
                self.args.append(step[1])
            else:
                self.actions.append(step)
                self.args.append(None)

    def run(self, input_data):
        """
        Run the pipeline step by step

        Parameters
        ----------
        input_data : any
            Input of the pipeline

        Returns
        -------
        output : any
            Output of the pipeline
        """
        output = input_data
        for action, args in zip(self.actions, self.args):
            if args is not None:
                output = action(output, args)
            else:
                output = action(output)
        return output


class DataLoader:
    def __call__(self, file_path):
        """
        Load data from a file

        Parameters
        ----------
        file_path : str
            Path of the file

        Return
        ----------
        data : any
            Data from file

        """
        with io.open(file_path, 'rb') as data_file:
            data = pickle.load(data_file)
        return data


class GraphConstructor:
    def __call__(self, contents, *args):
        """
        Create a WikiGraph

        Parameters
        ----------
        contents : array of strings
            Graph content
        args : tuple
            Arguments for graph creation
        """
        graph = WikiGraph(contents, *args)
        return graph


class ClusterConstructor:
    def __call__(self, graph):
        """
        Construct graph's cluster

        Parameters
        ----------
        graph : WikiGraph
            Graph from which we get the cluster
        """
        clusters = graph.clusters()
        return clusters


class Observer:
    def __call__(self, observed_object):
        """
        Print any data and return it

        Parameters
        ----------
        observed_object : any
           Printed object
        """
        print(observed_object)
        return observed_object
