import logging
import numpy as numpy
from itertools import islice

from .data_indexer import DataIndexer
from .dataset import TextDataset

logger = logging.getLogger(__name__)

class DataManager():
    """docstring for DataManager"""
    def __init__(self, instance_type):
        self.data_indexer = DataIndexer()
        self.instance_type = instance_type
        self.max_sentence_lengths = {}

    @staticmethod
    def batch_generator(instance_generator, batch_size):
        instance_generator = instance_generator()
        batched_instances = list(islice(instance_generator, batch_size))
        while batched_instances:
            flattened_inputs, flattened_targets = ([ins[0] for ins in batched_instances],
                                                    [ins[1] for ins in batched_instances])
            batch_inputs = tuple(map(np.array, tuple(zip(*flattened_inputs))))
            batch_targets = tuple(map(np.array, tuple(zip(*flattened_targets))))       
            yield batch_inputs, batch_targets
            batched_instances = list(islice(instance_generator, batch_size))
    
    def get_train_from_file(self, filenames, min_count=1,
                            max_instances=None,
                            max_sentence_lengths=None,
                            pad=True,
                            mode='word'):
        train_dataset = TextDataset.read_from_file(filenames,instance_class) 

        train_size = len(training_dataset.instances)
        self.data_indexer.fit(train_dataset,min_count=min_count)
        logger.info("Indexing dataset")
        indexed_train_dataset = train_dataset.to_indexed(self.data_indexer)
        train_max_lengths = indexed_train_dataset.max_lengths()
        def _get_train_generator():
            for instance in indexed_train_dataset.instances:
                if pad:
                    indexed_instance.pad(train_max_lengths)
                inputs, labels = indexed_instance.as_train(mode=mode)
                yield inputs,labels
            return _get_train_generator, train_size

    def get_valid_from_file(self,filenames,max_instances=None,
                            max_sentence_lengths=None,
                            pad=True,
                            mode='word'):
        valid_dataset = TextDataset.read_from_file(filenames)
        valid_size = len(valid_dataset.instances)
        indexed_valid_dataset = valid_dataset.to_indexed(self.data_indexer)
        valid_max_lengths = valid_dataset.max_lengths()
        def _get_valid_generator():
            for instance in indexed_valid_dataset.instances:
                if pad:
                    instance.pad(valid_max_lengths)
                inputs, labels = instance.as_train(mode=mode)
                yield inputs,labels
            return _get_valid_generator, valid_size

    def get_test_from_file(self, filenames,
                            max_sentence_lengths=None,
                            pad=True,
                            mode='word'):
        test_dataset = TextDataset.read_from_file(filenames)
        indexed_test_dataset = test_dataset.to_indexed()
        test_size = len(indexed_test_dataset.instances)
        test_max_lengths = indexed_test_dataset.max_lengths()
        def _get_test_generator():
            for instance in indexed_test_dataset:
                if pad:
                    instance.pad(test_max_lengths)
                inputs, labels = instance.as_test(mode=mode)
                yield inputs, labels
            return _get_test_generator, test_size
















