import argparse
import csv
import logging
import re
import os


logger = logging.getLogger(__name__)

def main():
    argparser = argparse.ArgumentParser(description=('Removing newlines and delete consequent blankspace'))
    argparser.add_argument('input_path', type=str,
                                        help=('The path of raw dataset'))
    argparser.add_argument('output_folder',type=str,
                                        help=('The folder the cleaned file be written to'))
    config = argparser.parse_args()
    logger.info("Reading csv at {}".format(config.input_path))
    logger.info('cleaning')
    cleaned_rows=[]
    with open(config.input_path) as f:
        reader = csv.reader(f)
        reader.__next__()
        for row in reader:
            cleaned_row = []
            for item in row:
                no_newlines = re.sub(r'\n', ' ', item)
                new_item = re.sub(r'\s+',' ', no_newlines)
                cleaned_row.append(new_item)
        cleaned_rows.append(cleaned_row)
    input_file = os.path.basename(config.input_path)
    input_filename, input_ext = os.path.splitext(input_file)
    output_path = os.path.join(config.output_folder,
                                input_filename+'_cleaned'+input_ext)
    logger.info('Writing output to {}'.format(output_path))
    with open(output_path,'w') as f:
        writer = csv.writer(f,quoting=csv.QUOTE_ALL)
        writer.writerows(cleaned_rows)

if __name__ == "__main__":
    logging.basicConfig(format=("%(asctime)s - %(levelname)s - "
                                "%(name)s - %(message)s"),
                        level=logging.INFO)
    main()
