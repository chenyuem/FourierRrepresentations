import numpy as np
import itertools
# import fourier_basics as fb
# import inference as inf
# import optimize as opt
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import svm
# from scipy.linalg import hadamard
import utils_valid
from collections import Counter
from numpy.linalg import matrix_rank
import random

data = np.loadtxt("data/win95.csv", delimiter=",", skiprows=2)
# data = data - 1.0

train, test = train_test_split(data, test_size=0.2, random_state=43)

m, n = train.shape
N = 2**n

k = 11
p = n - k
print p

# compute idx needed
idx_need = utils_valid.idx_sort(train, m)

# build A in Ax = 0 (mod 2)
A = utils_valid.build_A(p,n,idx_need)
print A.shape

# Gaussian Elimination
A_ge, Perm = utils_valid.gaussianElimination(A)
print A_ge.shape

# Solutions 
H = utils_valid.solutionH(A_ge, Perm)
print H.shape

random.seed(959)

if H.shape[1] > k:
	arr = np.arange(H.shape[1])
	np.random.shuffle(arr)
	H = H[:,arr[:k]]

# 2^k vecotrs set 0 -> 1
zero_base = np.where(np.sum(H,axis=1) == 0)
H[zero_base,:] = 1

# Compute positions in Fourier domain 
posF = utils_valid.posInFourier(H)
print H
print posF
print posF.shape

# import IPython
# IPython.embed()

# # For any pos, find bases
# pos = posF.astype('int8')
# bases, perm_b = utils_valid.gaussianElimination(pos)
# perm_b.reverse()
# B = bases.T
# if perm_b != []:
# 	for pair in perm_b:
# 		i,j = pair
# 		row_temp = B[i,:] + 0
# 		B[i,:] = B[j,:] + 0
# 		B[j,:] = row_temp + 0
# print B
# 
# # Rank "importance" of bases
# c_b = pos.dot(B) % 2
# print c_b
# bin = c_b >= 1
# bin_acc = np.sum(bin, axis=0)
# r_b = np.argsort(bin_acc)
# print r_b
# 
# # Choose pos for validity
# B_chosen = B[:,r_b[-k:]]
# print 'base_chosen'
# print B_chosen
# posValid = utils_valid.posInFourier(B_chosen)
# print 'posValid'
# print posValid

# Compute 2^k coefficients
coeff = utils_valid.coefficients(posF, train)
print coeff
print coeff.shape

# pause('input')

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

# import IPython
# IPython.embed()

# plt.figure()
# plt.plot(acc[:,0],acc[:,1],"o")
# plt.plot([0.63,0.97],[0.63,0.97])
# plt.xlabel("Fourier")
# plt.ylabel("SVM")
# plt.bar(range(n), acc[:,0], acc[:,1])
# plt.show()

