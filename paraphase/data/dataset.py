import codecs
import itertools
import logging

from tqdm import tqdm

from .instances.instance import Instance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Dataset:
    def __init__(self, instances):
       self.instances = instances

    def truncate(self, size):
        if not isinstance(size, int):
            raise ValueError("Expected size to be type int, found {} of type "
                             "{}".format(size, type(size)))
        if size < 1:
            raise ValueError("size must be at least 1, found {}".format(size))
        if len(self.instances) <= size:
            return self
        return self.__class__(self.instances[:size])

class TextDataset(Dataset):
    def __init__(self,instances):
        super(TextDataset,self).__init__(instances)

    def to_indexed(self, data_indexer):
        indexed_instances = [instance.to_indexed_instance(data_indexer) for
                             instance in tqdm(self.instances)]
        return IndexedDataset(indexed_instances)

    @staticmethod
    def read_from_file(filenames, instance_class):
        logger.info("Reading files {}.".format(filenames)) 
        lines = [x.strip() for filename in filenames
                 for x in tqdm(codecs.open(filename,
                                           "r", "utf-8").readlines())]
        return TextDataset.read_from_lines(lines, instance_class)

    @staticmethod
    def read_from_lines(lines, instance_class):
        logger.info("Creating list of {} instances from "
                    "list of lines.".format(instance_class))
        instances = [instance_class.read_from_line(line) for line in tqdm(lines)]
        return TextDataset(instances)


class IndexedDataset(Dataset):

    def __init__(self, instances):
        super(IndexedDataset, self).__init__(instances)

    def max_lengths(self):
        max_lengths = {}
        lengths = [instance.get_max_length() for instance in self.instances]
        if not lengths:
            return max_lengths
        for key in lengths[0]:
            max_lengths[key] = max(x[key] if key in x else 0 for x in lengths)
        return max_lengths

    def pad_instances(self, max_lengths=None):
        lengths_to_use = {}
        for key in instance_max_lengths:
            if max_lengths and max_lengths[key] is not None:
                lengths_to_use[key] = max_lengths[key]
            else:
                lengths_to_use[key] = instance_max_lengths[key]
        logger.info("Padding instances to length: %s",
                    str(lengths_to_use))
        for instance in tqdm(self.instances):
            instance.pad(lengths_to_use)

    def as_training_data(self, mode="word"):
        inputs = []
        labels = []
        instances = self.instances
        for instance in instances:
            instance_inputs, label = instance.as_train(mode=mode)
            inputs.append(instance_inputs)
            labels.append(label)
        return inputs, labels

    def as_testing_data(self, mode="word"):
        inputs = []
        instances = self.instances
        for instance in instances:
            instance_inputs, _ = instance.as_test(mode=mode)
            inputs.append(instance_inputs)
        return inputs, []
