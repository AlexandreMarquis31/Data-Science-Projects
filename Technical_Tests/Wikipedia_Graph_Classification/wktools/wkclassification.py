import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.metrics import Recall
import numpy as np
from wktools.wkutils import get_tokens


class WikiClassifier(keras.Model):
    """
    Wikipedia's page classification model

    Parameters
    ----------
    data_transformer : WikiDataTransformer
        Object used to transform Input/Output data, also contains input's details (max_size, vocab_size, topics_size)
    embedding_dim : int
        The word embedding dimension
    """
    def __init__(self, data_transformer, embedding_dim=500):
        super(WikiClassifier, self).__init__()
        self.data_transformer = data_transformer
        self.input_layer = Input(data_transformer.max_size)
        self.embedding = Embedding(data_transformer.vocab_size, embedding_dim)
        self.avg_pool = GlobalAveragePooling1D()
        self.dense1 = Dense(128, activation="relu")
        self.dropout2 = Dropout(0.3)
        self.dense2 = Dense(embedding_dim, activation="relu")
        self.classifier = Dense(data_transformer.topics_size, activation="sigmoid")
        self.out = self.call(self.input_layer)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.avg_pool(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.classifier(x)
        return x

    def compile(self, loss='categorical_crossentropy', optimizer='adam', metrics=None):
        """
        Compile the model with most appropriate params

        Parameters
        ----------
        loss : tensorflow.keras.losses or str
            Loss of the model
        optimizer : tensorflow.keras.optimizers or str
            Optimizer of the model
        metrics : tensorflow.keras.metrics or none
            Metric used to measure efficiency of the model
        """
        if metrics is None:
            metrics = [Recall()]
        metrics = metrics if metrics else [Recall()]
        Model.compile(self, loss=loss, optimizer=optimizer, metrics=metrics)

    def predict(self, content):
        """
        Predict a category with our model

        Parameters
        ----------
        content : str
            Wikipedia page's source code

        Returns
        -------
        topic : str
            Predicted topic class
        """
        x_input = self.data_transformer.transform_x(content)
        pred = self(np.array([x_input]))
        topic = self.data_transformer.inverse_transform_y(pred)
        return topic


class WikiDataTransformer:
    """
    Object for easy data/dataset manipulation

    Parameters
    ----------
    tokens_set : set of str
        The vocabulary

    topics : set of str
        Topics of wiki page

    max_size : int
        Maximal size for wikipedia page
    """
    def __init__(self, tokens_set, topics, max_size=1000):
        tokens = tokens_set
        tokens.add('<unknown>')  # token used when a word is not in the vocabulary
        tokens.add('<pad>')  # token used to fill a page when it is smaller than max_size
        self.dictionary = dict(zip(tokens, range(len(tokens))))
        self.vocab_size = len(self.dictionary)
        self.topics = list(topics)
        self.topics_size = len(list(topics))
        self.max_size = max_size

    def transform_split(self, input_x, input_y, train_size, test_size, batch_size=64, shuffle=True):
        """
        Create Train/Test/Validation datasets usable by WikiClassifier

        Parameters
        ----------
        input_x : list of str
            List of Wikipedia page's source code
        input_y : list of str
            Topics corresponding to the Wikipedia pages
        train_size : float
            Size of the train dataset (between 0 and 1)
        test_size : float
            Size of the test dataset (between 0 and 1)
        batch_size : int
            Size of dataset batch
        shuffle : boolean
            Boolean that indicate if we should shuffle the data

        Returns
        -------
        train_split : tensorflow.python.data.ops.dataset_ops.BatchDataset
            Train dataset
        test_split : tensorflow.python.data.ops.dataset_ops.BatchDataset
            Test dataset
        val_split : tensorflow.python.data.ops.dataset_ops.BatchDataset
            Validation dataset
        """
        x, y = self.transform(input_x, input_y)
        train_split, test_split, val_split = self.split(x, y, train_size, test_size, batch_size, shuffle)
        return train_split, test_split, val_split

    @staticmethod
    def split(x, y, train_size, test_size, batch_size=64, shuffle=True):
        """
        Split the data into 3 datasets

        Parameters
        ----------
        x : list
            Input
        y : list
            Labels
        train_size : float
            Size of the train dataset (between 0 and 1)
        test_size : float
            Size of the test dataset (between 0 and 1)
        batch_size : int
            Size of dataset batch
        shuffle : boolean
            Boolean that indicate if we should shuffle the data

        Returns
        -------
        train_split : tensorflow.python.data.ops.dataset_ops.BatchDataset
            Train dataset
        test_split : tensorflow.python.data.ops.dataset_ops.BatchDataset
            Test dataset
        val_split : tensorflow.python.data.ops.dataset_ops.BatchDataset
            Validation dataset
        """
        size = len(x)
        train_size = int(train_size * size)
        test_size = int(test_size * size)
        # val_size =  1 - (train_size + test_size)

        if shuffle:
            p = np.random.permutation(size)
            x, y = x[p], y[p]

        full_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset = full_dataset.take(train_size).batch(batch_size)
        other_dataset = full_dataset.skip(train_size)
        val_dataset = other_dataset.skip(test_size).batch(batch_size)
        test_dataset = other_dataset.take(test_size).batch(batch_size)

        return train_dataset, test_dataset, val_dataset

    def transform(self, input_x, input_y):
        """
        Transform the data in order to be usable by WikiClassifier

        Parameters
        ----------
        input_x : list of str
            List of Wikipedia source code page
        input_y : list of str
            Corresponding topics

        Returns
        -------
        x : list of list of int
            Transformed X (index-in-word-dictionary encoded, padded, truncated)
        y : list of list of int
            Ont-Hot-encoded topics
        """
        data_count = len(input_x)
        x = np.empty((data_count, self.max_size))
        for k, content in enumerate(input_x):
            x[k] = self.transform_x(content)
        y = self.transform_y(input_y)
        return x, y

    def transform_x(self, input_x):
        """
        Pad, truncate, index-in-word-dictionary encode input

        Parameters
        ----------
        input_x : list of str
            Wikipedia page source code

        Returns
        -------
        x : list of int
            Transformed input
        """
        x = np.full(self.max_size, self.dictionary['<pad>'])
        tokens = get_tokens(input_x)
        for i in range(min(len(tokens), self.max_size)):
            x[i] = self.dictionary.get(tokens[i], self.dictionary['<unknown>'])
        return x

    def transform_y(self, input_y):
        """
        Encode (One-Hot-encoder) topics

        Parameters
        ----------
        input_y : list of string
            Topics

        Returns
        -------
        y : np.ndarray
            Encoded topics
        """
        ordinal_y = np.asarray([self.topics.index(topic) for topic in input_y])  # ordinal encoding
        y = np.eye(self.topics_size)[ordinal_y]  # one-hot-encoding
        return y

    def inverse_transform_y(self, input_y):
        """
        Recover topic name from one-hot-encoding prediction

        Parameters
        ----------
        input_y : np.ndarray
            Encoded topic

        Returns
        -------
        y : str
            Decoded topic
        """
        best_pred = np.argmax(input_y)
        y = self.topics[best_pred]
        return y
