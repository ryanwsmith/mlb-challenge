"""
Usage: python create_model_datas.py sample_file_path

       sample_file_path: full or relative path to the file containing the combined corpus data[required]

Example: python create_model_datas.py corpus/combined_samples.txt
"""

import sys
import math
import csv

from MLB.ml import SampleCorpus
from MLB.ml import FeatureExtractor

TRAIN_SET_PERCENT = 0.7

def create_model_data(samples_path):
    sc = SampleCorpus(balanced_training=False)
    sc._get_samples_from_file(samples_path)

    training_data = []
    for labeled_sample in sc.labeled_samples:
        bin_b64 = labeled_sample[1]
        target_label = labeled_sample[0]
        fe = FeatureExtractor(bin_b64, [target_label, ], normalize_features=False)

        data = fe.feature_vector
        data.append(target_label)
        training_data.append(data)

    with open('mlb_training_set.csv', 'wb') as train_csvfile:
        with open('mlb_testing_set.csv', 'wb') as test_csvfile:
            train_writer = csv.writer(train_csvfile)
            test_writer = csv.writer(test_csvfile)

            train_size = int(math.floor(len(training_data)*TRAIN_SET_PERCENT))
            for data in training_data[:train_size]:
                train_writer.writerow(data)
            for data in training_data[train_size:]:
                test_writer.writerow(data)


if __name__ == "__main__":
    try:
        samples_path = sys.argv[1]
    except:
        print(__doc__)
        sys.exit(-1)

    create_model_data(samples_path)