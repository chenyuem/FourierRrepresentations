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

k = 8

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

pos = pos.astype('int8')

bases, perm_b = utils_valid.gaussianElimination(pos)
perm_b.reverse()
B = bases.T
if perm_b != []:
	for pair in perm_b:
		i,j = pair
		row_temp = B[i,:] + 0
		B[i,:] = B[j,:] + 0
		B[j,:] = row_temp + 0
print B

# Rank "importance" of bases
c_b = pos.dot(B) % 2
print c_b
bin = c_b >= 1
bin_acc = np.sum(bin, axis=0)
r_b = np.argsort(bin_acc)
print r_b

# Choose pos for validity
B_chosen = B[:,r_b[-k:]]
zero_base = np.where(np.sum(B_chosen,axis=1) == 0)
B_chosen[zero_base,:] = 1
# B_chosen[0,0] = 1
# B_chosen[1,1] = 1
# B_chosen[2,2] = 1
# B_chosen[0:n-k,:] = 1
print 'base_chosen'
print B_chosen

# zero_base = np.sum(B_chosen,axis=1) == 0
posValid = utils_valid.posInFourier(B_chosen)

# import IPython
# IPython.embed()

print 'posValid'
print posValid

posF = posValid

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
		# print (p0,p1)
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

