"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.
"""

import sys
import json
import numpy as np


def load_vocab(filename_vocab):
    """Loads and returns the mapping of word index to actual word string in the vocabulary.

    Args:
        filename_vocab (string): filepath.
    Returns:
        dictionary: int (index) to string (word) in the vocabulary.
    """
    vocab = dict()
    with open(filename_vocab, 'r') as fi:
        for line in fi.readlines():
            ind_word = line.strip().split()
            if len(ind_word) >= 2:
                vocab[int(ind_word[0])] = " ".join(ind_word[1:])
            else:
                print(ind_word)
    return vocab


def load_docs(filename_features, filename_labels):
    """Loads and returns features of all documents as well as their corresponding labels

    Args:
        filename_features (string): filepath to features.
        filename_labels (string): filepath to labels.
    Returns:
        X: A list of documents. Each document is a list of (word id, frequency) tuples, where
           each tuple specifies how many times the word with the word id appears in the document.
        Y: A list of integers ground-truth labels of documents in X.
    """
    X = []
    doc = []
    with open(filename_features, 'r') as fi:
        for line in fi.readlines():
            # Each document ends with '#'
            if '#' in line.strip():
                X.append(doc)
                doc = []
            else:
                freq = line.strip().split()
                if len(freq) == 2:
                    doc.append((int(freq[0]), int(freq[1])))
    Y = []
    with open(filename_labels, 'r') as fi:
        for y in fi.readlines():
            line = y.strip().split()
            Y.append(int(line[1]))
    return X, Y


def nb_train_smoothing(X, Y, vocab_size, num_classes, alpha):
    """Computes Pr(class) and Pr(word | class) given each document's label and word frequencies.

    Args:
        X: A list of N documents. Each document is a list of (word id, frequency) tuples, where
            each tuple specifies how many times the word with the word id appears in the document.
        Y: A list of N integers corresponding ground-truth labels of documents in X.
        vocab_size: a scalar corresponding to the vocabulary size or the number of words in the vocabulary.
        num_classes: a scalar corresponding to the number of classes.
        alpha: a scalar corresponding to smoothing parameter.
    Returns:
        class_prob: A numpy array of shape (num_classes, 1), where
                    class_prob[class] is the probability value Pr(class) after smoothing.
        class_word_prob: A numpy array of shape (num_classes, vocab_size), where
                    class_word_prob[class, word] is the probability value Pr(word | class) after smoothing.
    """
    class_prob = np.zeros((num_classes, 1))
    class_word_prob = np.zeros((num_classes, vocab_size))
    ###################################################
    # Q8.2 Edit here
    ###################################################

    totalword = 0
    map = {}
    numberY = {}
    numberY[0] = 0
    numberY[1] = 0
    map[0] = {}
    map[1] = {}
    totalzeronum = 0
    totalonenum = 0

    for i in range(len(Y)):
        numberY[Y[i]] = numberY.get(Y[i]) + 1
        listinx = X[i]
        tempvaluedic = map.get(Y[i])
        for j in range(len(listinx)):
            if not listinx[j][0] in tempvaluedic:
                tempvaluedic[listinx[j][0]] = 0
            tempvaluedic[listinx[j][0]] = tempvaluedic.get(listinx[j][0]) + listinx[j][1]
            totalword += listinx[j][1]
            if Y[i] == 0:
                totalzeronum += listinx[j][1]
            else:
                totalonenum += listinx[j][1]
        map[Y[i]] = tempvaluedic

    for key in map.keys():
        class_prob[key, 0] = numberY[key] * 1.0 / len(Y)
        for i in range(vocab_size):

            if map.get(key).get(i) == None:
                map.get(key)[i] = alpha

            if key == 0:
                class_word_prob[key, i] = map.get(key).get(i) / totalzeronum
            else:
                class_word_prob[key, i] = map.get(key).get(i) / totalonenum

    return class_prob, class_word_prob


def nb_train(X, Y, vocab_size, num_classes):
    """Computes Pr(class) and Pr(word | class) given each document's label and word frequencies.

    Args:
        X: A list of N documents. Each document is a list of (word id, frequency) tuples, where
            each tuple specifies how many times the word with the word id appears in the document.
        Y: A list of N integers corresponding ground-truth labels of documents in X.
        vocab_size: a scalar corresponding to the vocabulary size or the number of words in the vocabulary.
        num_classes: a scalar corresponding to the number of classes.
    Returns:
        class_prob: A numpy array of shape (num_classes, 1), where
                    class_prob[class] is the probability value Pr(class).
        class_word_prob: A numpy array of shape (num_classes, vocab_size), where
                    class_word_prob[class, word] is the probability value Pr(word | class).
    """
    class_prob = np.zeros((num_classes, 1))
    class_word_prob = np.zeros((num_classes, vocab_size))
    ###################################################
    # Q8.1 Edit here
    ###################################################
    totalword = 0
    map = {}
    numberY = {}
    numberY[0] = 0
    numberY[1] = 0
    map[0] = {}
    map[1] = {}
    totalzeronum = 0
    totalonenum = 0

    for i in range(len(Y)):
        numberY[Y[i]] = numberY.get(Y[i])+1
        listinx = X[i]
        tempvaluedic = map.get(Y[i])
        for j in range(len(listinx)):
            if not listinx[j][0] in tempvaluedic:
                tempvaluedic[listinx[j][0]] = 0
            tempvaluedic[listinx[j][0]] = tempvaluedic.get(listinx[j][0]) + listinx[j][1]
            totalword += listinx[j][1]
            if Y[i] == 0:
                totalzeronum += listinx[j][1]
            else:
                totalonenum += listinx[j][1]
        map[Y[i]] = tempvaluedic

    for key in map.keys():
        class_prob[key, 0] = numberY[key]*1.0 / len(Y)
        for i in range(vocab_size):
            if map.get(0).get(i) == None and map.get(1).get(i) == None:
                class_word_prob[0, i] = -1
                class_word_prob[1, i] = -1
                continue

            if map.get(key).get(i) == None:
                class_word_prob[key, i] = 0
            else:
                if key == 0:
                    class_word_prob[key, i] = map.get(key).get(i) / totalzeronum
                else:
                    class_word_prob[key, i] = map.get(key).get(i) / totalonenum

    return class_prob, class_word_prob


def nb_predict(X, class_prob, class_word_prob):
    """Predicts the class of each document in X, using Pr(class) and Pr(word | class).

    Args:
        X: A list of N documents. Each document is a list of (word id, frequency) tuples, where
           each tuple specifies how many times the word with the word id appears in the document.
        class_prob: A numpy array of shape (num_classes, 1), where
                    class_prob[class] is the probability value Pr(class).
        class_word_prob: A numpy array of shape (num_classes, vocab_size), where
                    class_word_prob[class, word] is the probability value Pr(word | class).
    Returns:
        Ypred: A list of N integers corresponding to predicted labels of documents in X.
               0 for ham and 1 for spam.
               If two classes are equally likely, predict 1 (spam).
    """
    Ypred = []
    ###################################################
    # Q8.1 Edit here
    ###################################################
    for i in range(len(X)):
        listinx = X[i]
        tempp = []
        for indexclass in range(len(class_prob)):
            p = calculatelog(class_prob[indexclass])
            for j in range(len(listinx)):
                if class_word_prob[indexclass, listinx[j][0]] == -1:
                    continue
                p += listinx[j][1] * calculatelog(class_word_prob[indexclass, listinx[j][0]])

            tempp.append(p)

        if(tempp[0] > tempp[1]):
            Ypred.append(0)
        else:
            Ypred.append(1)

    return Ypred

def calculatelog(x):
    if x == 0:
        return -1e50
    else:
        return np.log(x)

def compute_acc(Ypred, Ytrue):
    """Computes the accuracy given predicted and ground-truth labels.

    Args:
        Ypred: A list of N integers corresponding to predicted labels.
        Ytrue: A list of N integers corresponding to ground-truth labels.
    Returns:
        Accuracy in [0,1] of Ypred.
    """
    if len(Ypred) != len(Ytrue):
        return 0.0
    return np.mean(np.array(Ypred) == np.array(Ytrue))


# Arguments from command line
ftrain_feat = sys.argv[1]
ftrain_lab = sys.argv[2]
fval_feat = sys.argv[3]
fval_lab = sys.argv[4]
fstest_feat = sys.argv[5]
fstest_lab = sys.argv[6]
fvocab = sys.argv[7]
is_smooth = int(sys.argv[8]) == 1

# Load documents and vocab
print('Loading data...')
vocab = load_vocab(fvocab)
vocab_size = max([id for id in vocab]) + 1
Xtrain, Ytrain = load_docs(ftrain_feat, ftrain_lab)
Xval, Yval = load_docs(fval_feat, fval_lab)
Xstest, Ystest = load_docs(fstest_feat, fstest_lab)
print('Finished.')
num_classes = len(set([y for y in Ytrain]))

accs = []
if not is_smooth:
    # Q8.1
    # Train a naive Bayes model without smoothing.
    print('\n\nTraining...')
    class_prob, class_word_prob = nb_train(Xtrain, Ytrain, vocab_size, num_classes)
    print('Classifying emails...')
    # Predict labels of the documents based on class probability distribution and class-specific word distribution.
    Ytrain_pred = nb_predict(Xtrain, class_prob, class_word_prob)
    Yval_pred = nb_predict(Xval, class_prob, class_word_prob)
    Ystest_pred = nb_predict(Xstest, class_prob, class_word_prob)
    # Compute the accuracies of predicted labels.
    train_acc = 100*compute_acc(Ytrain_pred, Ytrain)
    val_acc = 100*compute_acc(Yval_pred, Yval)
    test_acc = 100*compute_acc(Ystest_pred, Ystest)
    print('Results')
    print('Training accuracy:     ', '{:.3f}'.format(train_acc), '%')
    print('Validation accuracy:   ', '{:.3f}'.format(val_acc), '%')
    print('Sampled test accuracy: ', '{:.3f}'.format(test_acc), '%')
    accs.append(train_acc)
    accs.append(val_acc)
    accs.append(test_acc)
    with open('nb1.json', 'w') as f_json:
        json.dump(accs, f_json)
else:
    # Q8.2
    for alpha in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
        # Train a naive Bayes model without smoothing for each smoothing parameter alpha.
        print('\n\nTraining...')
        class_prob, class_word_prob = nb_train_smoothing(Xtrain, Ytrain, vocab_size, num_classes, alpha)
        print('Classifying emails...')
        # Predict labels of the documents based on class probability distribution and class-specific word distribution.
        Ytrain_pred = nb_predict(Xtrain, class_prob, class_word_prob)
        Yval_pred = nb_predict(Xval, class_prob, class_word_prob)
        Ystest_pred = nb_predict(Xstest, class_prob, class_word_prob)
        # Compute the accuracies of predicted labels.
        train_acc = 100 * compute_acc(Ytrain_pred, Ytrain)
        val_acc = 100 * compute_acc(Yval_pred, Yval)
        test_acc = 100 * compute_acc(Ystest_pred, Ystest)
        print('Results with smoothing alpha =', alpha)
        print('Training accuracy:     ', '{:.3f}'.format(train_acc), '%')
        print('Validation accuracy:   ', '{:.3f}'.format(val_acc), '%')
        print('Sampled test accuracy: ', '{:.3f}'.format(test_acc), '%')
        accs.append(train_acc)
        accs.append(val_acc)
        accs.append(test_acc)
        with open('nb2.json', 'w') as f_json:
            json.dump(accs, f_json)