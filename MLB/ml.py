import base64
import math

class SampleCorpus(object):
    def __init__(self, balanced_training=True):
        self.balanced_training=balanced_training
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

        if self.balanced_training:
            self.training_set = self._balance_sample_set(self.labeled_samples[0:training_set_size])
        else:
            self.training_set = self.labeled_samples[0:training_set_size]
        self.validation_set = self.labeled_samples[training_set_size:num_samples]

    def _balance_sample_set(self, sample_set):
        #key: target/label,
        #value: list of bin64
        samples_by_label = {}

        for sample in sample_set:
            sample_target = sample[0]
            sample_bin_b64 = sample[1]

            if sample_target in samples_by_label:
                samples_by_label[sample_target].append(sample_bin_b64)
            else:
                samples_by_label[sample_target]= [sample_bin_b64,]

        #get the count of the target with the least number of samples in the sample set
        min_sample_count = min([len(sample_list) for sample_list in samples_by_label.values()])

        #create sample set with equal number of samples for each target
        balanced_sample_set = []

        for target, sample_list in samples_by_label.items():
            for sample_bin_b64 in sample_list[:min_sample_count]:
                balanced_sample_set.append((target, sample_bin_b64))

        return balanced_sample_set

class FeatureExtractor(object):
    def __init__(self, bin_b64, targets, normalize_features=True):
        """
        Initialize and create a feature vector for the provided binary sample.

        :param bin_b64: base64 encoded representation of the binary sample
        :type bin_b64: str

        :param targets: list of possible targets
        :type targets: list (str)

        :param normalize_features: determines whether feature values are absolute or normalized to a value between 0 and 1
        :type normalize_features: bool
        """
        # options
        self.normalize_features = normalize_features

        # internal data
        self._bin_b64 = bin_b64
        self._bin_byte_array = self._convert_b64_to_byte_array(self._bin_b64)
        self._targets = targets
        self._byte_val_count = {}

        # public data
        self.bin_len = self._get_bin_length()
        self.trailing_zero_count = self._get_trailing_zero_count()
        self.longest_zero_run_count = self._get_longest_zero_run_count()
        self.feature_vector = self._get_feature_vector()

    def _convert_b64_to_byte_array(self, b64_bin):
        """
        Convert the base64 encoded representation of the binary sample into a byte array containing the integer value of each byte

        :param b64_bin: base64 encoded representation of the binary sample
        :type b64_bin: str

        :return: a byte array containg the integer values of each byte in the binary sample
        :rtype: bytearray
        """
        bin_hex_string = base64.b64decode(b64_bin)
        bin_byte_array = bytearray(bin_hex_string)
        return bin_byte_array

    def _get_bin_length(self):
        """
            Calculate the length of the binary sample
            Note: it appears to always be 64, but this may not be guaranteed

            :return: length of the binary sample
            :rtype: int
        """
        return len(self._bin_byte_array)

    def _get_trailing_zero_count(self):
        """
            Calculate the number of trailing zeros at the end of the binary sample
            Note: since all samples are 64 bytes, trailing zeros could indicate padding for variable-length ISAs (if zero-padding is required/used)

            :return: the largest number of consecutive zeros in the binary sample
            :rtype: int
        """
        count = 0
        for i in range(len(self._bin_byte_array) - 1, -1, -1):
            if self._bin_byte_array[i] == 0:
                count += 1
            else:
                return count

    def _get_longest_zero_run_count(self):
        """
        Calculate the longest consecutive run of zeros
        Note: zero runs may indicate word size for multibyte zero constants

        :return: the largest number of consecutive zeros in the binary sample
        :rtype: int
        """
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
        """
        Count the occurance of common constants as an indicator for endian-ness

        Note: common constants could indicate endianess (e.g. mips vs mipsel), though small sample sizes may make them tough to find

        :return: A list of features containing the counts of 'interesting' 1-byte constants
        :return: list (int)
        """
        endian_indicator_counts = {0xfffe: 0,  # -2 BE
                                   0xfeff: 0,  # -2 LE
                                   0x0001: 0,  # 1 BE
                                   0x0100: 0,  # 1 LE
                                   0x0002: 0,  # 2 BE
                                   0x0200: 0,  # 2 LE
                                   0x0400: 0,  # 4 LE
                                   0x0004: 0,  # 4 BE
                                   0x0800: 0,  # 8 LE
                                   0x0008: 0,  # 8 BE
                                   }

        # count occurances of common 16-bit constants
        for i in range(len(self._bin_byte_array) - 1):
            byte_val = self._bin_byte_array[i]
            next_byte_val = self._bin_byte_array[i + 1]

            if byte_val == 0xff and next_byte_val == 0xfe:  # -2 BE
                endian_indicator_counts[0xfffe] += 1
            elif byte_val == 0xfe and next_byte_val == 0xff:  # -2 LE
                endian_indicator_counts[0xfeff] += 1
            elif byte_val == 0x00 and next_byte_val == 0x01:  # 1 BE
                endian_indicator_counts[0x0001] += 1
            elif byte_val == 0x01 and next_byte_val == 0x00:  # 1 LE
                endian_indicator_counts[0x0100] += 1
            elif byte_val == 0x00 and next_byte_val == 0x02:  # 2 BE
                endian_indicator_counts[0x0002] += 1
            elif byte_val == 0x02 and next_byte_val == 0x00:  # 2 LE
                endian_indicator_counts[0x0200] += 1
            elif byte_val == 0x04 and next_byte_val == 0x00:  # 4 LE
                endian_indicator_counts[0x0400] += 1
            elif byte_val == 0x00 and next_byte_val == 0x04:  # 4 BE
                endian_indicator_counts[0x0004] += 1
            elif byte_val == 0x08 and next_byte_val == 0x00:  # 8 LE
                endian_indicator_counts[0x0800] += 1
            elif byte_val == 0x00 and next_byte_val == 0x08:  # 8 BE
                endian_indicator_counts[0x0008] += 1

        endian_indicator_vector = endian_indicator_counts.values()
        return endian_indicator_vector

    def _get_feature_vector(self):
        """
        Proccess the byte array of the binary sample to extract a list of 'interesting' features which may correlate to the corresponding ISA

        :return: the feacture vector for the provided binary sample (mixed type)
        :rtype: list
        """
        fv = []
        for i in range(256):
            fv.append(0)

        # freq count
        for byte in self._bin_byte_array:
            fv[int(byte)] += 1

        # normalize byte counts
        if self.normalize_features:
            for i in range(256):
                fv[i] = (fv[i] * 1.0) / len(self._bin_byte_array)

        # add trailing zero feature
        trail_count = self._get_trailing_zero_count()

        if self.normalize_features:
            fv.append((trail_count * 1.0) / len(self._bin_byte_array))
        else:
            fv.append(trail_count * 1.0)

        # add longest zero run feature
        run_count = self._get_longest_zero_run_count()
        if self.normalize_features:
            fv.append((run_count * 1.0) / len(self._bin_byte_array))
        else:
            fv.append(run_count * 1.0)

        # add endian indicator counts
        endian_indicator_counts = self._get_endian_val_counts()

        if self.normalize_features:
            for i in range(len(endian_indicator_counts)):
                endian_indicator_counts[i] = (endian_indicator_counts[i] * 1.0) / (len(self._bin_byte_array) / 2)

        fv.extend(endian_indicator_counts)

        return fv
