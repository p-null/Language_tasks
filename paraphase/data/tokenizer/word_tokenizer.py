import nltk 

class NLTKWordTokenizer():
    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence.lower())

    def tokenize_for_indexer(self, sentence):
        word = nltk.word_tokenize(sentence)
        char = list(map(list, word))
        return {'word':word, 'char':char}

