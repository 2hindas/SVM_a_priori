import numpy as np
import pandas as pd
import scipy.ndimage as im
from PIL import Image

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((0, 255))

from src.SupportVectorMachine import SupportVectorMachine
from src.Kernels import JitteredKernel
from pprint import pprint
pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)

px = 20
num_f = px ** 2


def view(arr):
    trans = np.uint8((arr.reshape(20, 20) + 1) / 2.0 * 255)
    img = Image.fromarray(trans, 'L')
    img.show()


train = pd.read_csv("../data/USPS_train.csv", delim_whitespace=True)
test = pd.read_csv("../data/USPS_test.csv", delim_whitespace=True)

transformations = [(1, 0),  # D
                   (-1, 0),  # U
                   (0, 1),  # R
                   (0, -1),  # L
                   (1, 1),  # RD
                   (1, -1),  # LD
                   (-1, 1),  # RU
                   (-1, -1)]  # LU

train_features = train.iloc[:, 1:].to_numpy()
train_targets = train.iloc[:, 0].to_numpy()
test_features = test.iloc[:, 1:].to_numpy()
test_targets = test.iloc[:, 0].to_numpy()

print("Data has been read")

train_features = np.pad(train_features.reshape((7290, 16, 16)), ((0, 0), (2, 2), (2, 2)),
                        'constant',
                        constant_values=(-1)).reshape((7290, num_f))
test_features = np.pad(test_features.reshape((2006, 16, 16)), ((0, 0), (2, 2), (2, 2)), 'constant',
                       constant_values=(-1)).reshape((2006, num_f))





print("Training...")

start = timer()
s = SupportVectorMachine(train_features, train_targets, test_features, test_targets)

o_svm = svm.SVC(cache_size=800)
o_svm.fit(train_features, train_targets)
time1 = timer()
print("First svm has been trained in:", time1-start)

feat = train_features[o_svm.support_]
targ = train_targets[o_svm.support_]

a = s.rotate_SV(feat, -5, 0, 5)
b = s.rotate_SV(feat, 0, 5, 5)
c = s.translate_SV(feat, transformations[0:1], 1, 1)
d = s.translate_SV(feat, transformations[1:2], 1, 1)
e = s.translate_SV(feat, transformations[2:3], 1, 1)
print(test_features.shape)
translated_test = s.translate_SV(test_features, transformations[2:3], 1, 1)
test_features = np.vstack((test_features, translated_test))
test_targets = np.append(test_targets, test_targets)

print(translated_test.shape)
print(test_features.shape)
# f = s.translate_SV(feat, transformations[3:4], 1, 1)
# g = s.translate_SV(feat, transformations[4:5], 2, 2)
# h = s.translate_SV(feat, transformations[5:6], 2, 2)
# i = s.translate_SV(feat, transformations[6:7], 2, 2)
# j = s.translate_SV(feat, transformations[7:8], 2, 2)
time2 = timer()
print("Data transformed in:", time2-time1)

og_svm = svm.SVC(cache_size=800)
og_svm.fit(feat, targ)
# a_svm = svm.SVC(cache_size=800)
# a_svm.fit(a, targ)
# b_svm = svm.SVC(cache_size=800)
# b_svm.fit(b, targ)
# c_svm = svm.SVC(cache_size=800)
# c_svm.fit(c, targ)
# d_svm = svm.SVC(cache_size=800)
# d_svm.fit(d, targ)
e_svm = svm.SVC(cache_size=800)
e_svm.fit(e, targ)

# f_svm = svm.SVC(cache_size=800)
# f_svm.fit(f, targ)
# g_svm = svm.SVC(cache_size=800)
# g_svm.fit(g, targ)
# h_svm = svm.SVC(cache_size=800)
# h_svm.fit(h, targ)
# i_svm = svm.SVC(cache_size=800)
# i_svm.fit(i, targ)
# j_svm = svm.SVC(cache_size=800)
# j_svm.fit(j, targ)
time3 = timer()
print("Other svm's trained in:", time3-time2)


# running = o_svm.decision_function(test_features)
#
# running = np.add(running, a_svm.decision_function(test_features))
# running = np.add(running, b_svm.decision_function(test_features))
# running = np.add(running, c_svm.decision_function(test_features))
# running = np.add(running, d_svm.decision_function(test_features))
# running = np.add(running, e_svm.decision_function(test_features))
# 6.281000000000006

# running = o_svm.predict(test_features).reshape(-1, 1)
# running = np.hstack((running, a_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, b_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, c_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, d_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, e_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, f_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, g_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, h_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, i_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, j_svm.predict(test_features).reshape(-1, 1)))

# predictions = np.argmax(running, axis=1)
pred1 = og_svm.predict(test_features)
pred2 = e_svm.predict(test_features)
diff1 = pred1 - test_targets
diff2 = pred2 - test_targets
wrong1 = np.nonzero(diff1)
wrong2 = np.nonzero(diff2)
correct2 = np.where(diff2 == 0)[0]

indices = np.intersect1d(wrong1, correct2)

probs_0 = og_svm.decision_function(test_features)
probs_e = e_svm.decision_function(test_features)

classes_0 = np.argmax(probs_0, axis=1)
classes_e = np.argmax(probs_e, axis=1)

max_0 = probs_0[np.arange(len(probs_0)), np.argmax(probs_0, axis=1)]
max_e = probs_e[np.arange(len(probs_e)), np.argmax(probs_e, axis=1)]

sum_0 = max_0 * np.sum(np.square(probs_0 - max_0.reshape(-1, 1)), axis=1)
sum_e = max_e * np.sum(np.square(probs_e - max_e.reshape(-1, 1)), axis=1)

vals = sum_0.reshape(-1, 1)
vals = np.hstack((vals, sum_e.reshape(-1, 1)))

classes = classes_0.reshape(-1, 1)
classes = np.hstack((classes, classes_e.reshape(-1, 1)))

pred3 = classes[np.arange(len(classes)), np.argmax(vals, axis=1)]



# exit(0)





# m = np.maximum(probs_0, probs_e)
# classes = np.argmax(m, axis=1)
print("Normal: ", accuracy_score(test_targets, pred1))
print("Translated: ", accuracy_score(test_targets, pred2))
print("Probs: ", accuracy_score(test_targets, pred3))

print(probs_0[indices])
pprint(pred1[indices])
pprint(pred3[indices])
pprint(test_targets[indices])
pprint(pred2[indices])
print(probs_e[indices])

# try distances between highest point and others



"""
VSV 19.507 seconds
Virtual Features 36.627 seconds

91.719 seconds error 3.84

train -> transformations[0:4], 1, 1 -> train -> transformations[0:4], 1, 1 ->
272.195 seconds error 3.44

train -> transformations[0:4], 1, 1 -> train(0.5) -> transformations[0:4], 1, 1 -> train(0.5)
5.28 -> 4.14 -> 3.54 138 seconds

train -> rotate_SV(-5, 5, 5) -> train -> translate_SV(transformations[0:4], 1, 2) -> train C=10
4.74 -> 4.34 -> 3.49 167.377 seconds

train -> rotate_SV(-5, 5, 5) -> train -> translate_SV(transformations[0:4], 1, 1) -> train C=10
4.74 -> 4.34 -> 3.09 60.698 seconds

train -> rotate_SV(-5, 5, 5) -> train -> translate_SV(transformations[0:8], 1, 1) -> train C=10
4.74 -> 4.34 -> 2.94 145.828 seconds

train -> rotate_SV(-10, 10, 5) -> train -> translate_SV(transformations[0:8], 1, 1) -> train C=10
4.74 -> 4.29 -> 2.94 303.533 seconds

train -> rotate_SV(-5, 5, 5) -> train -> translate [0:4], 1, 1 & [4:8], 2, 2 -> train C=10
4.74 -> 4.34 -> 2.89 187.674 seconds

train -> rotate_SV(-5, 5, 5) -> train -> translate [0:4], 1, 1 & [4:8], 2, 2 -> train C=1
5.28 -> 4.64 -> 3.04 298.330 seconds



train -> rotate_SV(-5, 5, 5), translate [0:4], 1, 1 & [4:8], 2, 2 -> train C=1
5.28 -> 3.64 134.226 seconds




WITH 2 PX PADDINGS

0.9466600199401795 (no jitter)                      error = 5.333998006 %
0.9491525423728814 (1 px jitter, 4 dir)             error = 5.0847457627 %
0.9516450648055832 (1 px jitter, 8 dir)             error = 4.8354935194 %
0.9561316051844466 (1 & 2 px jitter, 4 dir)         error = 4.386839482 %
0.9571286141575274 (1 & 2 px jitter, 8 dir)         error = 4.287138584 %

0.9471585244267199 (1 & 2 px jitter, UD)
0.9511465603190429 (1 & 2 px jitter, diagonals)
0.9526420737786641 (1 & 2 px jitter, LR)            error = 4.735792622 %
0.9561316051844466 -> 0.9536390827517448 (first 1&2 LRUD J, second 1&2 Diagonals V)
0.9526420737786641 -> 0.9541375872382851 (first 1&2 LR J, second 1&2 Diagonals J)
0.9526420737786641 -> 0.9596211365902293 (first 1&2 LR J, second 1&2 Diagonals V) error = 4,037886341
0.9511465603190429 -> 0.9501495513459621 (first 1&2 Diagonals J, second 1&2 LR V)

"""


# [[355   0   2   0   1   0   0   0   1   0]
#  [  0 255   0   0   5   0   3   1   0   0]
#  [  2   0 184   2   3   2   0   1   4   0]
#  [  2   0   3 148   0   8   0   0   5   0]
#  [  0   1   3   0 188   1   1   1   1   4]
#  [  2   0   0   3   1 150   0   0   1   3]
#  [  3   0   3   0   2   2 159   0   1   0]
#  [  0   0   1   0   5   1   0 137   1   2]
#  [  2   0   2   2   0   2   1   1 155   1]
#  [  0   0   0   1   6   0   0   0   0 169]]
# [[339   0   6   1   1   1   8   0   3   0]
#  [  0 237   0   0   4   0  14   0   9   0]
#  [  1   0 190   1   2   0   0   1   3   0]
#  [  1   0  12 122   1  21   0   0   8   1]
#  [  0   2   6   0 188   0   1   0   0   3]
#  [  2   1   3   1   1 148   0   0   2   2]
#  [  0   0   4   0   2   3 160   0   1   0]
#  [  0   1   4   0   8   0   0 126   7   1]
#  [  3   0   5   0   1   6   1   0 150   0]
#  [  0   3   0   0  15   1   0   2  10 145]]