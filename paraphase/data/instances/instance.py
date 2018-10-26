from ..tokenizer.word_tokenizer import NLTKWordTokenizer

class Instance:
    """docstring for Instance"""
    def __init__(self, label):
        self.label = label

class TextInstance(Instance):
    """docstring for TextInstance"""
    def __init__(self, label=None, tokenizer=None):
        super(TextInstance, self).__init__(label)
        if not tokenizer:
            self.tokenizer = NLTKWordTokenizer
        else:
            self.tokenizer = tokenizer

    def text2index(self, tokenized_sentence, data_indexer):
        #tokenized_sentence = self.tokenizer.tokenize_for_indexer(sentence)
        indexed_text_word = [data_indexer.get_word_index(word) 
                            for word in tokenized_sentence['word']] 
        indexed_text_char = []
        for word in tokenized_sentence['char']:
            indexed_text_char.append([data_indexer.get_char_index(char)
                                for char in word])
        return (indexed_text_word, indexed_text_char)

    #def words(self):
    #    raise NotImplementedError

    def to_indexed_instance(self, data_indexer):
        raise NotImplementedError

    @classmethod
    def read_from_line(cls,line):
        raise RuntiemError("can't be read from line")

    def get_pair_dict(self):
        raise NotImplementedError


class IndexedInstance(Instance):
    """docstring for IndexedInstance"""
    def get_max_length(self, arg):
        raise NotImplementedError

    def as_train(self):
        raise NotImplementedError

    def as_test(self):
        raise NotImplementedError

    def pad_instance(self, max_length):
        raise NotImplementedError

    @staticmethod
    def pad_sentence(sentence,
                    length,
                    default =lambda: 0,
                    left_align = True):
        if left_align:
            truncated = sequencce[:length]
        else:
            truncated = sequencce[-length:]
        if len(truncated)<length:
            padding_seq = [default()]*(length-len(truncated))
            if left_align:
                truncated.extend(padding_seq)
            else:
                padding_seq.extend(truncated)
        return truncated


    
        