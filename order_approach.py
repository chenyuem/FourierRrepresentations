import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import svm
import utils_valid
from collections import Counter
from numpy.linalg import matrix_rank
import random

data = np.loadtxt("data/win95.csv", delimiter=",", skiprows=2)
# data = data - 1.0

train, test = train_test_split(data, test_size=0.2, random_state=43)

m, n = train.shape
N = 2**n

# k = 5

pos_size = 1 + n + n*(n-1)/2
pos = np.zeros([pos_size,n]).astype('int8')
pos[1:n+1,:] = np.eye(n)
pointer = n+1
for i in range(1, n+1):
	for j in range(i+1, n+1):
		pos[pointer,:] = pos[i,:] ^ pos[j,:]
		pointer += 1

# Compute coefficients in Fourier domain
print pos

posF = pos
coeff = utils_valid.coefficients(posF, train)
print coeff
print coeff.shape

# Inference 1|n-1
acc = []
n_i = n
for j in range(n_i):
	print "index ", j
	y_pre0 = np.zeros([test.shape[0]])
	for i in range(test.shape[0]):
		assign0 = 2 * test[i,:] - 1
		assign1 = 2 * test[i,:] - 1
		assign0[j] = -1
		assign1[j] = 1
		assign0 = assign0[:n_i]
		assign1 = assign1[:n_i]
		p0 = posF[:,:n_i].dot(assign0).dot(coeff)
		p1 = posF[:,:n_i].dot(assign1).dot(coeff)
		if p0 == p1:
			y_pre0[i] = int(round(np.random.rand(1)[0]))
		else:
			y_pre0[i] = np.argmax([p0,p1])
	pred_ind = np.setdiff1d(range(n_i),j)
	clf = svm.SVC()
	clf.fit(train[:,pred_ind],train[:,j])
	y_pre = clf.predict(test[:,pred_ind])
	acc.append([sum(y_pre0==test[:,j]) * 1.0 / len(y_pre0), sum(y_pre==test[:,j]) * 1.0 / len(y_pre)])
	print "accuracy of Fourier", sum(y_pre0==test[:,j]) * 1.0 / len(y_pre0) 
	print "accuracy of SVM", sum(y_pre==test[:,j]) * 1.0 / len(y_pre)
	# if j == 2:
	# 	import IPython
	# 	IPython.embed()

acc = np.array(acc)