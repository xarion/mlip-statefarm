
import sys
import time

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.cross_validation import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV, ElasticNetCV, LarsCV, LassoCV, LassoLarsCV, \
    MultiTaskElasticNetCV, \
    MultiTaskLassoCV, OrthogonalMatchingPursuitCV, RidgeClassifierCV
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data_root = '/Users/erdicalli/dev/workspace/statefarm-data/'
submission_root = '/Users/erdicalli/dev/workspace/statefarm/submissions/'
# data_root = '/home/ml0501/statefarm/'
# submission_root = '/home/ml0501/erdi/submissions/'

import h5py

# LAYERS = ['fc6']
LAYERS = ['fc7']
# LAYERS = ['fc6', 'fc7']
file_identifier = "".join(LAYERS)
LAYER_SIZE = 4096
DATA_POINTS = LAYER_SIZE * len(LAYERS)
num_classes = 10


def convert_label_to_class_array(str_label):
    return [int(str_label[1:])]


f = h5py.File(data_root + 'train_image_' + file_identifier + 'features.h5', 'r')
labels = [convert_label_to_class_array(x) for x in f['class']]
features = np.copy(f["feature"])
f.close()

names = [
    "SVC(kernel='rbf', probability=True)",
    "SVC(kernel='linear', probability=True)",
    "SVC(kernel='sigmoid', probability=True)",
    "SVC(kernel='poly', probability=True, degree=3)",
    "SVC(kernel='poly', probability=True, degree=4)",
    "SVC(kernel='poly', probability=True, degree=5)",
    "DecisionTreeClassifier()",
    "KNeighborsClassifier()",
    "GaussianNB()",
    "RandomForestClassifier()",
    "AdaBoostClassifier()",
    "QuadraticDiscriminantAnalysis()",
    "LinearDiscriminantAnalysis()",
    "ElasticNetCV()",
    "LarsCV()",
    "LassoCV()",
    "LassoLarsCV()",
    "LogisticRegressionCV()",
    "MultiTaskElasticNetCV()",
    "MultiTaskLassoCV()",
    "OrthogonalMatchingPursuitCV()",
    "RidgeClassifierCV()"
]

output_file_names = [
    "SVCRBF",
    "SVCLINEAR",
    "SVCSIGMOID",
    "SVCPOLYD3",
    "SVCPOLYD4",
    "SVCPOLYD5",
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
    "GaussianNB",
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "QuadraticDiscriminantAnalysis",
    "LinearDiscriminantAnalysis",
    "ElasticNetCV",
    "LarsCV",
    "LassoCV",
    "LassoLarsCV",
    "LogisticRegressionCV",
    "MultiTaskElasticNetCV",
    "MultiTaskLassoCV",
    "OrthogonalMatchingPursuitCV",
    "RidgeClassifierCV"
]

classifiers = [
    SVC(kernel="rbf", probability=True),
    SVC(kernel='linear', probability=True),
    SVC(kernel='sigmoid', probability=True),
    SVC(kernel='poly', probability=True, degree=3),
    SVC(kernel='poly', probability=True, degree=4),
    SVC(kernel='poly', probability=True, degree=5),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(),
    ElasticNetCV(max_iter=10000),
    LarsCV(),
    LassoCV(max_iter=10000),
    LassoLarsCV(),
    LogisticRegressionCV(),
    MultiTaskElasticNetCV(),
    MultiTaskLassoCV(),
    OrthogonalMatchingPursuitCV(),
    RidgeClassifierCV()
]
algorithm = 17
if len(sys.argv) > 1:
    algorithm = int(sys.argv[1])

name = names[algorithm]
clf = classifiers[algorithm]
output_file_name = output_file_names[algorithm] + file_identifier

t = time.time()
random_state = np.random.RandomState(0)
print "Fitting classifier " + name
classifier = OneVsRestClassifier(clf, n_jobs=2)
labels = MultiLabelBinarizer().fit_transform(labels)

# this is what happens when sklearn doesn't support cross validation for class probabilities.
kf = KFold(features.shape[0], n_folds=10)
predictions = None
print "Running 10-Fold Cross Validation"
step = 0
for train_index, test_index in kf:
    print "Running Fold: %d" % (step)
    fold_train_features, fold_test_features = features[train_index, :], features[test_index, :]
    fold_train_labels, fold_test_labels = labels[train_index, :], labels[test_index, :]
    cross_validation_classifier = clone(classifier)
    cross_validation_classifier.fit(fold_train_features, fold_train_labels)
    fold_predictions = cross_validation_classifier.predict_proba(fold_test_features)
    if predictions is None:
        predictions = fold_predictions
    else:
        predictions = np.concatenate((predictions, fold_predictions), axis=0)
    step += 1

print "Classifier: " + name
print "Time passed: ", "{0:.1f}".format(time.time() - t), "sec"
print "Log Loss: ", log_loss(labels, predictions)

print "Fitting Classifier"
classifier.fit(features, labels)

print "Calculating Predictions..."

del features
del labels

f = h5py.File(data_root + 'test_image_' + file_identifier + 'features.h5', 'r')
test_features = np.copy(f["feature"])
test_photo_ids = np.copy(f["photo_id"])
f.close()

t = time.time()
predicted_class_probabilities = classifier.predict_proba(test_features)

print "Calculated Predictions... Time passed: ", "{0:.1f}".format(time.time() - t), "sec"
print "Writing predictions to output file"
index = 1
df = pd.DataFrame(columns=['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
for i in zip(test_photo_ids, predicted_class_probabilities):
    result = [i[0]]
    result.extend(i[1])
    df.loc[index] = result
    index += 1

with open(submission_root + "results" + output_file_name + ".csv", 'w') as f:
    df.to_csv(f, index=False, header=True, float_format='%e')
