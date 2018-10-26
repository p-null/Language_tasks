import logging

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)  

class EmbeddingManager():
    """
    An EmbeddingManager takes a DataIndexer fit on a train dataset,
    and produces an embedding matrix with pretrained embedding files.
    """
    def __init__(self, data_indexer):
        if not data_indexer.is_fit:
            raise ValueError("DataIndexer must first be fit on input data.")
        self.data_indexer = data_indexer

    @staticmethod
    def initialize_random_matrix(shape, scale=0.05, seed=0):
        numpy_rng = np.random.RandomState(seed)
        return numpy_rng.uniform(low=-scale, high=scale, size=shape)

    def get_embedding_matrix(self, embedding_dim,
                             pretrained_embeddings_file_path=None,
                             pretrained_embeddings_dict=None,
                             namespace="words"):
        embeddings_from_file = {}
        if pretrained_embeddings_file_path:
            logger.info("Reading pretrained embeddings file from {}".format(
                            pretrained_embeddings_file_path))
            with open(pretrained_embeddings_file_path) as embedding_file:
                for line in tqdm(embedding_file):
                    fields = line.strip().split(" ")
                    if len(fields) - 1 <= 1:
                        raise ValueError("Found embedding size of 1; "
                                         "do you have a header?")
                    if embedding_dim != len(fields) - 1:
                        raise ValueError("Provided embedding_dim of {}, but "
                                         "file at pretrained_embeddings_"
                                         "file_path has embeddings of "
                                         "size {}".format(embedding_dim,
                                                          len(fields) - 1))
                    word = fields[0]
                    vector = np.array(fields[1:], dtype='float32')
                    embeddings_from_file[word] = vector

        if pretrained_embeddings_dict:
            embeddings_dict_dim = 0
            for word, vector in pretrained_embeddings_dict.items():
                if not embeddings_dict_dim:
                    embeddings_dict_dim = len(vector)
                if embeddings_dict_dim != len(vector):
                    raise ValueError("Found vectors of different lengths in "
                                     "the pretrained_embeddings_dict.")
            if embeddings_dict_dim != embedding_dim:
                raise ValueError("Provided embedding_dim of {}, but "
                                 "pretrained_embeddings_dict has embeddings "
                                 "of size {}".format(embedding_dim,
                                                     embeddings_dict_dim))

        vocab_size = self.data_indexer.get_vocab_size(namespace=namespace)
        # Build the embedding matrix
        embedding_matrix = self.initialize_random_matrix((vocab_size,
                                                          embedding_dim))
        for i in range(2, vocab_size):
            # Get the word corresponding to the index
            word = self.data_indexer.get_word_from_index(i)
            if (pretrained_embeddings_dict and
                    word in pretrained_embeddings_dict):
                embedding_matrix[i] = pretrained_embeddings_dict[word]
            else:
                if embeddings_from_file and word in embeddings_from_file:
                    embedding_matrix[i] = embeddings_from_file[word]
        return embedding_matrix
