#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.KernelCombinations import basic_combinations
import logging
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from src.SupportVectorMachine import SupportVectorMachine
import hickle as hkl

tf.enable_eager_execution()
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0)

directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

# ### Retrieve MNIST dataset

# In[23]:


gamma = 0

dataset = "MNIST"
X_train = hkl.load(f'../data/{dataset}/{dataset}_train_features.hkl')
Y_train = hkl.load(f'../data/{dataset}/{dataset}_train_labels.hkl')
X_test = hkl.load(f'../data/{dataset}/{dataset}_test_features.hkl')
Y_test = hkl.load(f'../data/{dataset}/{dataset}_test_labels.hkl')


def pooling_kernel(x_matrix, y_matrix):
    num_features = x_matrix.shape[1]
    sqrt_features = int(np.sqrt(num_features))
    x_length = x_matrix.shape[0]
    y_length = y_matrix.shape[0]

    gram_matrix = rbf_kernel(x_matrix, y_matrix, gamma=A_gamma)

    #     x_reshaped = x_matrix.reshape(x_length, sqrt_features, sqrt_features, 1)
    #     y_reshaped = y_matrix.reshape(y_length, sqrt_features, sqrt_features, 1)

    #     max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
    #                                          strides=1)
    #     filtered_x = max_pool(x_reshaped).numpy() \
    #         .reshape(x_length,
    #                  np.square(int(sqrt_features - 1)))
    #     filtered_y = max_pool(y_reshaped).numpy() \
    #         .reshape(y_length,
    #                  np.square(int(sqrt_features - 1)))

    #     pooled_matrix = rbf_kernel(filtered_x, filtered_y, gamma=B_gamma)

    #     max_pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
    #                                           strides=1)
    #     filtered_x_2 = max_pool2(x_reshaped).numpy() \
    #         .reshape(x_length,
    #                  np.square(int(sqrt_features - 2)))
    #     filtered_y_2 = max_pool2(y_reshaped).numpy() \
    #         .reshape(y_length,
    #                  np.square(int(sqrt_features - 2)))

    #     pooled_matrix_2 = rbf_kernel(filtered_x_2, filtered_y_2, gamma=C_gamma)

    return gram_matrix


# In[25]:


def test_error(model, features, labels):
    error = (1.0 - accuracy_score(labels, model.predict(features))) * 100
    return np.around(error, decimals=2)


# ### Run SVM with pooling kernel

# In[46]:


gammas = np.logspace(-1.8, -1, 15)
# gammas = np.logspace(-1.9, -0.65, 15)
# gammas = np.logspace(-3.5, -0.65, 15)
num_runs = 1

errors = np.zeros(len(gammas))
VSV_errors = np.zeros(len(gammas))

for rstate in range(0, num_runs):
    # Test the pooling kernel 

    current_errors = []
    for g in gammas:
        A_gamma = B_gamma = C_gamma = g
        model = svm.SVC(C=2, cache_size=50, kernel=pooling_kernel)
        model.fit(X_train, Y_train)
        current_errors.append(test_error(model, X_test, Y_test))
    errors = np.asarray(current_errors)

    # Test VSV method
#     current_errors = []
#     for g in gammas:
#         model = SupportVectorMachine(X_train, Y_train, X_test, Y_test, g)
#         model.train()
#         model.translate_SV(directions[0:8], 1, 1)
#         model.train()
#         current_errors.append(model.error())

#     VSV_errors += np.asarray(current_errors)

# VSV_errors = VSV_errors / num_runs
errors = errors / num_runs

# In[47]:


plt.clf()
plt.ylim(0, 100)
plt.plot(gammas, errors, label='Standard RBF')
# plt.plot(gammas, errors[1], label='Pooling 2x2')
# plt.plot(gammas, errors[2], label='Pooling 3x3')
# plt.plot(gammas, VSV_errors, label='VSV')

# plt.scatter(gammas[np.argmin(errors[0])], errors[0][np.argmin(errors[0])])
# plt.scatter(gammas[np.argmin(errors[1])], errors[1][np.argmin(errors[1])])
# plt.scatter(gammas[np.argmin(errors[2])], errors[2][np.argmin(errors[2])])
# plt.scatter(gammas[np.argmin(VSV_errors)], VSV_errors[np.argmin(VSV_errors)])

# plt.title(f"Artificial Lines dataset")
plt.legend(loc='upper right')
plt.xlabel("$\gamma$")
plt.ylabel("Generalization error")
print(gammas)
print(errors)
# print(VSV_errors)
plt.savefig("kernel-comparison.png", dpi=400)
# plt.show()

# In[48]:


for i in range(0, len(errors)):
    print(f"({gammas[i]}, {errors[i]})")

# print()

# for i in range(0, len(VSV_errors)):
#         print(f"({gammas[i]}, {VSV_errors[i]})")  


# In[49]:


print("Minimum errors:\n")

# VSV_gamma = gammas[np.argmin(VSV_errors)]
standard_gamma = gammas[np.argmin(errors)]
# pooling2_gamma = gammas[np.argmin(errors[1])]
# pooling3_gamma = gammas[np.argmin(errors[2])]

# optimal_gammas = [standard_gamma, pooling2_gamma, pooling3_gamma, VSV_gamma]

# print(f"VSV SVM: {VSV_errors[np.argmin(VSV_errors)]} with gamma {VSV_gamma}" )
print(f"Standard SVM: {errors[np.argmin(errors[0])]} with gamma {standard_gamma}")
# print(f"Pooling 2x2 SVM: {errors[1][np.argmin(errors[1])]} with gamma {pooling2_gamma}" )
# print(f"Pooling 3x3 SVM: {errors[2][np.argmin(errors[2])]} with gamma {pooling3_gamma}" )

exit()

# ### Plot learning curves

# In[150]:


index = np.argmin(errors)
A_gamma = B_gamma = C_gamma = gammas[index % len(gammas)]

samples = range(5, 1000, 50)
num_runs = 5

LC_errors = np.zeros((len(basic_combinations), len(samples)))
VSV_LC_errors = np.zeros(len(samples))

for rstate in range(0, num_runs):

    for i, num in enumerate(samples):

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=3 * n - num,
                                                            random_state=rstate)
        # Test the pooling kernel
        for j, f in enumerate(basic_combinations):
            A_gamma = B_gamma = C_gamma = optimal_gammas[j]
            combiner = f
            model = svm.SVC(C=2, cache_size=50, kernel=pooling_kernel)
            model.fit(X_train, Y_train)
            LC_errors[j][i] += test_error(model, X_test, Y_test)

        # Test VSV method
        model = SupportVectorMachine(X_train, Y_train, X_test, Y_test, VSV_gamma)
        model.train()
        model.translate_SV(directions[0:8], 1, 1)
        model.train()
        VSV_LC_errors[i] += model.error()

LC_errors = LC_errors / num_runs
VSV_LC_errors = VSV_LC_errors / num_runs

# In[185]:


print(LC_errors)
print(VSV_LC_errors)

# plt.clf()
# plt.ylim(0, 75)
# plt.plot(samples, LC_errors[0], label='Standard RBF')
# plt.plot(samples, LC_errors[1], label='Pooling 2x2')
# plt.plot(samples, LC_errors[2], label='Pooling 3x3')
# plt.plot(samples, VSV_LC_errors, label='VSV')
# # plt.title(f"Learning curve for artificial lines dataset,\n with {3*n} samples in total")
# plt.legend(loc='upper right')
# plt.xlabel("Number of training samples")
# plt.ylabel("Generalization error")
# plt.savefig("learning-curve.png", dpi=400)
# plt.show()

for e in LC_errors:
    for i in range(0, len(e)):
        print(f"({list(range(5, 1000, 50))[i]}, {e[i]})")

    print()

for i in range(0, len(VSV_LC_errors)):
    print(f"({list(range(5, 1000, 50))[i]}, {VSV_LC_errors[i]})")

# ### Performance of SVMs on noisy data

# In[167]:


gammas = np.logspace(-1.6, -0.65, 15)
num_runs = 5

noise_errors = np.zeros((len(basic_combinations), len(gammas)))
noise_VSV_errors = np.zeros(len(gammas))

for rstate in range(0, num_runs):
    X_train, X_test, Y_train, Y_test = train_test_split(X_noise, Y, test_size=test_ratio,
                                                        random_state=rstate)

    # Test the pooling kernel 
    for i, f in enumerate(basic_combinations):
        combiner = f
        current_errors = []
        for g in gammas:
            A_gamma = B_gamma = C_gamma = g
            model = svm.SVC(C=2, cache_size=50, kernel=pooling_kernel)
            model.fit(X_train, Y_train)
            current_errors.append(test_error(model, X_test, Y_test))
        noise_errors[i] += np.asarray(current_errors)

    # Test VSV method
    current_errors = []
    for g in gammas:
        model = SupportVectorMachine(X_train, Y_train, X_test, Y_test, g)
        model.train()
        model.translate_SV(directions[0:8], 1, 1)
        model.train()
        current_errors.append(model.error())

    noise_VSV_errors += np.asarray(current_errors)

noise_VSV_errors = noise_VSV_errors / num_runs
noise_errors = noise_errors / num_runs

# In[186]:


for error in noise_errors:
    for i in range(0, len(error)):
        print(f"({gammas[i]}, {error[i]})")

    print()

for i in range(0, len(noise_VSV_errors)):
    print(f"({gammas[i]}, {noise_VSV_errors[i]})")

# In[178]:

plt.clf()
plt.ylim(0, 70)
plt.plot(gammas, noise_errors[0], label='Standard RBF')
plt.plot(gammas, noise_errors[1], label='Pooling 2x2')
plt.plot(gammas, noise_errors[2], label='Pooling 3x3')
plt.plot(gammas, noise_VSV_errors, label='VSV')

VSV_gamma_noise = gammas[np.argmin(noise_VSV_errors)]
standard_gamma_noise = gammas[np.argmin(noise_errors[0])]
pooling2_gamma_noise = gammas[np.argmin(noise_errors[1])]
pooling3_gamma_noise = gammas[np.argmin(noise_errors[2])]

noise_optimal_gammas = [standard_gamma_noise, pooling2_gamma_noise, pooling3_gamma_noise,
                        VSV_gamma_noise]

# plt.title(f"Artificial Lines dataset")
plt.legend(loc='upper left')
plt.xlabel("$\gamma$")
plt.ylabel("Generalization error")
print(gammas)
print(noise_errors)
print(noise_VSV_errors)
plt.savefig("kernel-comparison-noise.png", dpi=400)
plt.show()

# In[176]:


print("Minimum noise errors:\n")

print(f"VSV SVM: {noise_VSV_errors[np.argmin(noise_VSV_errors)]} with gamma {VSV_gamma_noise}")
print(
    f"Standard SVM: {noise_errors[0][np.argmin(noise_errors[0])]} with gamma {standard_gamma_noise}")
print(
    f"Pooling 2x2 SVM: {noise_errors[1][np.argmin(noise_errors[1])]} with gamma {pooling2_gamma_noise}")
print(
    f"Pooling 3x3 SVM: {noise_errors[2][np.argmin(noise_errors[2])]} with gamma {pooling3_gamma_noise}")

# ### Generate learning curves for noisy data set

# In[169]:


index = np.argmin(noise_errors)
A_gamma = B_gamma = C_gamma = gammas[index % len(gammas)]
samples = range(5, 1050, 50)

num_runs = 5

LC_errors_noise = np.zeros((len(basic_combinations), len(samples)))
VSV_LC_errors_noise = np.zeros(len(samples))

for rstate in range(0, num_runs):

    for i, num in enumerate(samples):

        X_train, X_test, Y_train, Y_test = train_test_split(X_noise, Y, test_size=3 * n - num,
                                                            random_state=rstate)
        # Test the pooling kernel
        for j, f in enumerate(basic_combinations):
            A_gamma = B_gamma = C_gamma = noise_optimal_gammas[j]
            combiner = f
            model = svm.SVC(C=2, cache_size=50, kernel=pooling_kernel)
            model.fit(X_train, Y_train)
            LC_errors_noise[j][i] += test_error(model, X_test, Y_test)

        # Test VSV method
        model = SupportVectorMachine(X_train, Y_train, X_test, Y_test, VSV_gamma_noise)
        model.train()
        model.translate_SV(directions[0:8], 1, 1)
        model.train()
        VSV_LC_errors_noise[i] += model.error()

LC_errors_noise = LC_errors_noise / num_runs
VSV_LC_errors_noise = VSV_LC_errors_noise / num_runs

# In[174]:


plt.clf()
plt.ylim(0, 75)
plt.plot(samples, LC_errors_noise[0], label='Standard RBF')
plt.plot(samples, LC_errors_noise[1], label='Pooling 2x2')
plt.plot(samples, LC_errors_noise[2], label='Pooling 3x3')
plt.plot(samples, VSV_LC_errors_noise, label='VSV')
# plt.title(f"Learning curve for noisy artificial lines dataset,\n with {3*n} samples in total")
plt.legend(loc='upper right')
plt.xlabel("Number of training samples")
plt.ylabel("Generalization error")
plt.savefig("learning-curve-noise.png", dpi=400)
plt.show()
print(noise_optimal_gammas)
print(LC_errors_noise)
print(VSV_LC_errors_noise)

# In[188]:


for e in LC_errors_noise:
    for i in range(0, len(e)):
        print(f"({list(range(5, 1050, 50))[i]}, {e[i]})")

    print()

for i in range(0, len(VSV_LC_errors_noise)):
    print(f"({list(range(5, 1050, 50))[i]}, {VSV_LC_errors_noise[i]})")
