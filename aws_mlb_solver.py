"""
Usage: python aws_mlb_solver.py aws_ml_model_id

       aws_ml_model_id: ID of the AWS Machine Learning Model to use for predictions [required]

Example: python aws_mlb_solver.py ml-XXXXXXXXXXXXX
"""

import boto
import sys
import logging

from MLB.server import Server
from MLB.ml import FeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)


def create_record(feature_vector):
    """
    Create a prediction record in the format required by the specified AWS Machine Learning Model endpoint
    Note: apparently AWS does not care if you have extra parameters, so...
    TODO: unit test to ensure a that this produces a valid record that maches the schema for the AWS ML Model

    :param feature_vector: list of feature values for a given binary sample (mixed types)
    :type feature_vector: list

    :return: dictionary with keys: ML feature name (str) and values: feature value (str)
    :rtype: dict
    """
    var_format = "Var{0:0>3}"

    record = {}
    for i in range(len(feature_vector)):
        var_name = var_format.format(i + 1)  # ML Vars start at 001, doh!
        record[var_name] = str(feature_vector[i])

    return record


def predict(ml_model_id, record):
    """
    Query the AWS Machine Learning Model endpoint for a prediction given the provided feature values in the provided record.

    :param ml_model_id: The ID of the trained AWS Machine Learning Model: https://console.aws.amazon.com/machinelearning/home?region=us-east-1#/predictors
    :type ml_model_id: str

    :param record: dictionary with keys: ML feature name (str) and values: feature value (str)
    :type record: dict

    :return: AWS Machine Learning endpoint prediction for the given record
    :rtype: dict
    """
    ml = boto.connect_machinelearning()
    model = ml.get_ml_model(ml_model_id)
    endpoint = model.get('EndpointInfo', {}).get('EndpointUrl', '')
    if endpoint:
        prediction = ml.predict(ml_model_id, record, predict_endpoint=endpoint)
        return prediction


def solve_challenge(ml_model_id):
    """
        Solve the MLB challenge

        :param ml_model_id: The ID of the trained AWS Machine Learning Model: https://console.aws.amazon.com/machinelearning/home?region=us-east-1#/predictors
        :type ml_model_id: str
        """
    server = Server(log=LOG)

    attempt = 0
    current_streak = 0
    longest_streak = 0

    while True:
        attempt += 1
        LOG.info("ATTEMPT #: {}".format(str(attempt)))

        server.get()

        fe = FeatureExtractor(server.bin_b64, server.targets, normalize_features=False)
        record = create_record(fe.feature_vector)
        prediction = predict(ml_model_id, record)

        target = prediction['Prediction']['predictedLabel']
        if target not in server.targets:
            scores = prediction['Prediction']['predictedScores']

            for target in scores.keys():
                if target not in server.targets:
                    del scores[target]

            max_target = None
            max_score = -1

            for target, score in scores.items():
                if score > max_score:
                    max_target = target
                    max_score = score

            target = max_target

        LOG.debug("PREDICTION: {}".format(str(target)))
        server.post(target)

        if target == server.ans:
            LOG.debug('Win!')
            current_streak += 1
        else:
            LOG.debug('lose :(')
            if current_streak > longest_streak:
                longest_streak = current_streak
            current_streak = 0

        LOG.info("ACCURACY: {}".format(str(server.accuracy)))
        LOG.info("CURRENT STREAK: {}".format(str(current_streak)))
        LOG.info("LONGEST STREAK: {}".format(str(longest_streak)))

        if server.hash is not None:
            LOG.info("Winner!")
            LOG.info(server.hash)

            #stash the hash away for safe keeping
            f = open('mlb_hash.txt', 'w')
            f.write(server.hash)
            f.close()


if __name__ == "__main__":
    try:
        aws_ml_model_id = sys.argv[1]
    except Exception as e:
        print str(__doc__)
        sys.exit(-1)

    solve_challenge(aws_ml_model_id)
