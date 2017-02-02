# mlb-challenge

This is a solution the challenge: https://www.praetorian.com/challenges/machine-learning/index.html

The challenge is to train a machine learning model to correctly identify the Instruction Set Architecture for the provided binary samples (64 bytes)

## Solution Description

The core of this solution leverages the new AWS Machine Learning service (https://aws.amazon.com/machine-learning/)

While the may go a bit against the spirit of the challenge (if the intent was to implement low-level model primitives from scratch), but I feel it was appropos given the challenge starts by saying "Much of the work we do on a daily basis can be automated and classified by a machine, leaving us to focus on more interesting and challenging problems." ;)

Either way, it was a great opportunity to test the new AWS machine learning service, and so far I find it both quite useful and a great time savings :)

Most of the work was spent collecting samples, selecting features, extracting features, then training and testing the model.

## Feature Vector Description
* features 0-255
** This is a histogram of the occurance of byte values 0-255 in the binary sample.
** The hypothesis is that opcodes are common byte values, so a higher occurance of certain values may correlate to a particular ISA
*feature 256
** This is the number of trailing zeros at the end of the sample
** The hypothesis is since all samples are 64 bytes, trailing zeros could indicate padding for variable-length ISAs (if zero-padding is required/used)
* feature 257
** This is the length of the longest number of consecutive zeros in the sample
** The hypothes is that zero runs may indicate word size for multibyte zero constants
* feature features 258-267
** These are counts of common multibyte constants in both little endian and big endian representation
** The hypothesis is that this could identify the endian-ness of the ISA


## Solution Steps

1. Run build_corpus.py to collect labeled samples for a training set
2. Compbine all corpus files into a single file (if you ran build_corpus.py multiple times or with more than one thread)
3. Run create_model_data.py with the compbined sample you created in the previous step
4. [skipping lots of minor steps] train a new multi-class model with the training set created in the previous set in AWS (https://console.aws.amazon.com/machinelearning)
5. (optional) evaluate the model you created with the test set created in step #3.  We know by now that it works as long as it's provided enough training data, but we can still tinker with the feature vector and iterate if we like.
6. Create an endpoint for the model you've trained in AWS
7. Run aws_mlb_solver.py with the ID of the model you've trained in AWS
8. The hash _should_ drop out into a file "mlb_hash.txt" after a bit once it hits a 500 win streak (maybe a few hours)
