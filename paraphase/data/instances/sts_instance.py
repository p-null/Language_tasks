import csv
import itertools
import numpy as np
from copy import deepcopy
from overrides import overrides

from .instance import TextInstance, IndexedInstance
from .instance_word import IndexedWord

class QInstance(TextInstance):
    """docstring for QInstance"""
    def __init__(self,
                first_sentence,
                second_sentence,
                label):
        super(QInstance, self).__init__(label=label)
        self.first_sentence_tokenized = self.tokenizer.tokenize_for_indexer(first_sentence)
        self.second_sentence_tokenized = self.tokenizer.tokenize_for_indexer(second_sentence)

    def get_pair_dict(self):
        pair = deepcopy(self.first_sentence_tokenized)
        second  = deepcopy(self.second_sentence_tokenized)
        pair['char'] = list(itertools.chain.from_iterable(pair['char']))
        sec['char'] = list(itertools.chain.from_iterable(second['char']))
        for namespace in pair:
            pair[namespace].extend(second[namespace])
        return pair

    def read_from_line(self,line):
        fields = list(csv_reader(line))[0]
        if len(fields) == 6:
            _,_,_,first_sentence,second_sentence,label = fields
            label = int(lebel)
        elif len(fields) == 3:
            first_sentence,second_sentence,label = fields
            label = None
        else:
            raise RuntimeError('wrong format of this line{}'.format(fields))
        return cls(first_sentence,second_sentence,label)

    def to_indexed_instance(self,data_indexer):
        indexed_first_word, indexed_first_char = self.text2index(self.first_sentence_tokenized,data_indexer)
        indexed_second_word, indexed_second_char = self.text2index(self.second_sentence_tokenized,data_indexer)
        indexed_first_sentence = [IndexedWord(word, char) 
                                for word, char in zip(indexed_first_word,indexed_first_char)]
        indexed_second_sentence = [IndexedWord(word, char)
                                for word, chat in zip(indexed_second_word,indexed_first_char)]
        return IndexedQInstance(indexed_first_sentence, indexed_second_sentence)

class IndexedQInstance(IndexedInstance):
    """docstring for IndexedQInstance"""
    def __init__(self, indexed_first_sentence, indexed_second_sentence, label):
        super(IndexedQInstance, self).__init__(label)
        self.indexed_first_sentence = indexed_first_sentence
        self.indexed_second_sentence = indexed_second_sentence

    def get_word_indices(self):
        first_sentence_word_indices = [word.word_idx for word in indexed_first_sentence]
        second_sentence_word_indices = [word.word_idx for word in indexed_first_sentence]
        return (first_sentence_word_indices, second_sentence_word_indices)

    #def get_char_indices(self):
    def get_max_length(self):
        first_word_len = len(self.indexed_first_sentence)
        second_word_len = len(self.indexed_second_sentence)
        lengths = {'max_sentence_words':max(first_word_len,second_word_len)}
        return lengths

    def pad(self, max_length):
        max_num_words = max_length['max_sentence_words']
        self.indexed_first_sentence = self.pad_sentence(indexed_first_sentence,
                                                        max_num_words,
                                                        IndexedWord.padding_instance_word)
        self.indexed_second_sentence = self.pad_sentence(indexed_second_sentence,
                                                        max_num_words,
                                                        IndexedWord.padding_instance_word)

    def as_train(self,mode='word'):
        if self.label is None:
            raise ValueError("lable is None")
        if mode not in set(['word','char','word+char']):
            raise ValueError('mode error')
        if mode == 'word' or mode =='word+char':
            first_sentence_word_array = np.asarray([word.word_idx 
                                                    for word in self.indexed_first_sentence],
                                                    dtype='int32')
            second_sentence_word_array = np.asarray([word.word_idx 
                                                    for word in self.indexed_second_sentence],
                                                    dtype='int32')

        if mode == 'word':
            return ((first_sentence_word_array, second_sentence_word_array),(np.asarray(self.label),))


    def as_test(self, mode='word'):
        if mode not in set(['word','char','word+char']):
            raise ValueError('mode error')        
        if mode == 'word' or mode == 'word+char':
            first_sentence_word_array = np.asarray([word.word_idx for word in indexed_first_sentence])
            second_sentence_word_array = np.asarray([word.word_idx for word in indexed_second_sentence])

        if mode == 'word':
            return ((first_sentence_word_array, second_sentence_word_array),())