import os,sys
import base64
import math
from sklearn import tree

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class FeatureExtractor(object):
    def __init__(self, bin_b64, targets):
        self._bin_b64 = bin_b64
        self._bin_byte_array = self._convert_b64_to_byte_array(self._bin_b64)
        self._targets = targets

        self.bin_len = self._get_bin_length()
        self.trailing_zero_count = self._get_trailing_zero_count()
        self.longest_zero_run_count = self._get_longest_zero_run_count()

        self._byte_val_count = {}
        self.feature_vector = self._get_feature_vector()

    def _convert_b64_to_byte_array(self, b64_bin):
        bin_hex_string = base64.b64decode(b64_bin)
        bin_byte_array = bytearray(bin_hex_string)
        return bin_byte_array

    def _get_bin_length(self):
        return len(self._bin_byte_array)

    def _get_trailing_zero_count(self):
        count = 0
        for i in range(len(self._bin_byte_array)-1,-1,-1):
            if self._bin_byte_array[i] == 0:
                count += 1
            else:
                return count

    def _get_longest_zero_run_count(self):
        max_run_count = 0
        current_count = 0
        for byte_val in self._bin_byte_array:
            if byte_val == 0:
                current_count += 1
            else:
                if current_count > max_run_count:
                    max_run_count = current_count
                current_count = 0
        return max_run_count

    def _get_endian_val_counts(self):
        endian_indicator_counts = {0xfffe: 0,
                                   0xfeff: 0,
                                   0x0001: 0,
                                   0x0100: 0,
                                   0x0002: 0,
                                   0x0200: 0,
                                   0x0400: 0,
                                   0x0004: 0,
                                   0x0800: 0,
                                   0x0008: 0,
                                   }

        for i in range(len(self._bin_byte_array)-1):
            byte_val = self._bin_byte_array[i]
            next_byte_val = self._bin_byte_array[i+1]

            if byte_val == 0xff and next_byte_val == 0xfe:
                endian_indicator_counts[0xfffe]+=1
            elif byte_val == 0xfe and next_byte_val == 0xff:
                endian_indicator_counts[0xfeff] += 1
            elif byte_val == 0x00 and next_byte_val == 0x01:
                endian_indicator_counts[0x0001] += 1
            elif byte_val == 0x01 and next_byte_val == 0x00:
                endian_indicator_counts[0x0100] += 1
            elif byte_val == 0x00 and next_byte_val == 0x02:
                endian_indicator_counts[0x0002]+=1
            elif byte_val == 0x02 and next_byte_val == 0x00:
                endian_indicator_counts[0x0200] += 1
            elif byte_val == 0x04 and next_byte_val == 0x00:
                endian_indicator_counts[0x0400] += 1
            elif byte_val == 0x00 and next_byte_val == 0x04:
                endian_indicator_counts[0x0004] += 1
            elif byte_val == 0x08 and next_byte_val == 0x00:
                endian_indicator_counts[0x0800] += 1
            elif byte_val == 0x00 and next_byte_val == 0x08:
                endian_indicator_counts[0x0008] += 1

        endian_indicator_vector = endian_indicator_counts.values()

        for i in range(len(endian_indicator_vector)):
            endian_indicator_vector[i] = (endian_indicator_vector[i]*1.0)/(len(self._bin_byte_array)/2)

        return endian_indicator_vector


    def _get_feature_vector(self):
        fv = []
        for i in range(256):
            fv.append(0)

        #freq count
        for byte in self._bin_byte_array:
            fv[int(byte)] += 1

        #normalize byte counts
        for i in range(256):
            fv[i] = (fv[i]*1.0)/len(self._bin_byte_array)

        #add trailing zero feature
        trail_count = self._get_trailing_zero_count()
        fv.append((trail_count*1.0)/len(self._bin_byte_array))

        #add longest zero run feature
        run_count = self._get_longest_zero_run_count()
        fv.append((run_count * 1.0) / len(self._bin_byte_array))

        #add endian indicator counts
        endian_indicator_counts = self._get_endian_val_counts()
        fv.extend(endian_indicator_counts)

        return fv


class SampleCorpus(object):
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.labeled_samples = []
        self.training_set = []
        self.validation_set = []


    def _get_samples_from_file(self, sample_file_path):
        with open(sample_file_path) as sample_file:
            for sample_data in sample_file:
                sample_data_parts = sample_data.split(',')
                sample_target = sample_data_parts[0].strip()
                sample_bin_b64 = sample_data_parts[1].strip()
                self.labeled_samples.append((sample_target,sample_bin_b64))

    def _split_sample_set(self):
        train_set_percent = 0.8
        num_samples = len(self.labeled_samples)
        training_set_size = int(math.floor(train_set_percent*num_samples))

        self.training_set = self.labeled_samples[0:training_set_size]
        self.validation_set = self.labeled_samples[training_set_size:num_samples]

def solve():
    wins = 0
    attempts = 0
    best_streak = 0
    current_streak = 0

    corpus_path = '/Users/ryanw_smith/Code/personal/mlb-challenge/corpus'
    sc = SampleCorpus(corpus_path)

    target_values = {1: 'm68k',
                     2: 's390',
                     3: 'sh4',
                     4: 'powerpc',
                     5: 'alphaev56',
                     6: 'mipsel',
                     7: 'xtensa',
                     8: 'mips',
                     9: 'sparc',
                     10: 'avr',
                     11: 'arm',
                     12: 'x86_64'}

    target_values_by_name = dict((v, k) for k, v in target_values.iteritems())

    corpus_file_path = os.path.join(corpus_path, 'all.txt')
    sc._get_samples_from_file(corpus_file_path)
    sc._split_sample_set()

    training_vectors = []
    training_targets = []
    for training_sample in sc.training_set:
        bin_b64 = training_sample[1]
        target_label = training_sample[0]
        fe = FeatureExtractor(bin_b64, [target_label, ])

        training_vectors.append(fe.feature_vector)
        training_targets.append(target_values_by_name[target_label])

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(training_vectors, training_targets)
    # tree.export_graphviz(clf, out_file='tree.dot')


    for validation_sample in sc.validation_set:
        attempts += 1

        bin_b64 = validation_sample[1]
        target_label = validation_sample[0]
        fe = FeatureExtractor(bin_b64, [target_label, ])
        fe._get_endian_val_counts()

        target_guess = clf.predict([fe.feature_vector, ])[0]

        # print target_guess
        # print clf.predict_proba([fe.feature_vector,])
        # print "GUESS: {}, ACTUAL: {}".format(target_values[target_guess],target_label)

        if target_values[target_guess] == target_label:
            # print "SUCCESS!"
            wins += 1
            current_streak += 1
        else:
            print "FAIL!"
            print "GUESS: {}, ACTUAL: {}".format(target_values[target_guess], target_label)
            print clf.predict_proba([fe.feature_vector, ])
            print clf.predict_log_proba([fe.feature_vector, ])
            print clf.decision_path([fe.feature_vector, ])
            if current_streak > best_streak:
                best_streak = current_streak
            current_streak = 0

    accuracy = (wins * 1.0) / attempts
    print "ACCURACY: {}".format(accuracy)
    print "BEST STREAK: {}".format(best_streak)

if __name__ == "__main__":
    solve()