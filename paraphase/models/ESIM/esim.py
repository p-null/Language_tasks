from copy import deepcopy
from overrides import overrides
from tensorflow.contrib.rnn import LSTMCell
inport logging
import tensorflow as tf

from ..matching import bilater_matching
from ..base_tf_model import BaseTFModel 
from ..util.rnn import last_relevant_output

logger = logging.getLogger(__name__)

class BiMPM(BaseTFModel):
	def __init__(self, config_dict):
		config_dict = deepcopy(config_dict)
		mode = config_dict.pop('mode')
		super(BiMPM,self).__init__(mode=mode)

		self.word_vocab_size = config_dict.pop('word_vocab_size')
		self.word_embedding_dim = config_dict.pop('word_embedding_dim')
		self.word_embedding_matrix = config_dict.pop('word_embedding_matrix')
		self.char_vocab_size = config_dict.pop('char_vocab_size')
		self.char_embedding_dim = config_dict.pop('char_embedding_dim')
		self.char_embedding_matrix = config_dict.pop('char_embedding_matrix')
		
		self.char_rnn_hidden_size = config_dict.pop('char_rnn_hidden_size')
		self.context_rnn_hidden_size = config_dict.pop('context_rnn_hidden_size')

		self.embedding_trainable =  config_dict.pop('embedding_trainable')
		self.aggregation_rnn_hidden_size = config_dict.pop('aggregation_rnn_hidden_size')

	def _create_placeholder(self):
		