from collections import Counter, defaultdict
import logging

from .dataset import Dataset

import tqdm

logger = logging.getLogger(__name__)



class DataIndexer:
    def __init__(self):
        self._padding_token = '@PADDING@'
        self._oov_token = '@UNKOWN@'
        self.word_indices = defaultdict(default_dict)
        self.reverse_word_indices = defaultdict(default_dict_reverse())
        slef.is_fit = False
    def default_dict(self):
        return {self._padding_token:0,self._oov_token:1}
    def default_dict_reverse(self):
        return {0:self._oov_token,1:self._padding_token}

    def fit(self, dataset, min_count=1):
        logger.info("Fitting word dictionary with min count of %d", min_count)
        namespace_word_counts = defaultdict(Counter)
        for instance in tqdm.tqdm(dataset.instances):
            pair_dict = instance.get_pair_dict()
            for namespace in pair_dict:
                for word in pair_dict[namespace]:
                    namespace_word_counts[namespace].update([word]) #word is a string
        for namespace, word_counts in namespace_word_counts.items():
            sorted_word_counts = sorted(word_counts.items(),
                                        key=lambda pair: (-pair[1],
                                                          pair[0]))
            for word, count in sorted_word_counts:
                if count >= min_count:
                    self.add_word_to_index(word, namespace)
        self.is_fit = True

    def add_word_to_index(self, word, namespace="words"):
        if word not in self.word_indices[namespace]:
            index = len(self.word_indices[namespace])
            self.word_indices[namespace][word] = index
            self.reverse_word_indices[namespace][index] = word
            return index
        else:
            return self.word_indices[namespace][word]

    #def words_in_index(self, namespace="words"):

    def get_word_index(self, word, namespace="words"):
        if word in self.word_indices[namespace]:
            return self.word_indices[namespace][word]
        else:
            return self.word_indices[namespace][self._oov_token]

    def get_word_from_index(self, index, namespace="words"):
        return self.reverse_word_indices[namespace][index]

    def get_vocab_size(self, namespace="word"): 
        return len(self.word_indices[namespace])
