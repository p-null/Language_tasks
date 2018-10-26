import os 
import csv
import random
import logging
import argparse

logger = logging.getLogger(__name__)

def main():
	argparser = argparse.ArugumentParser(
		description=("Split a file into train and valid, given valid proportion")) 
	argparser.add_argument("valid_proportion", type=float)
	argparser.add_argument('input_path',type=str)
	argparser.add_argument('output_folder',type=str)

	config = argparser.parse_args()

	logger.info("Reading csv at {}".format(config.input_path))
	with open(config.input_path0) as f:
		reader = csv.reader(f)
		all_rows = list(reader)

	random.seed(0)
	random.shuffle(all_rows)
	