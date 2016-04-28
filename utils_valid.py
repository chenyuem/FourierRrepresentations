from collections import Counter
import numpy as np

def idx_sort(train, m):
	data_list = []
	for i in range(m):
		assign_temp = list(train[i,:].astype('int8'))
		assign_str = ' '.join(map(str, assign_temp))
		data_list.append(assign_str)
	data_counter = Counter(data_list)
	idx = data_counter.most_common()
	return idx

def build_A(p,n,idx_need):
	A = np.zeros([p,n])
	for i in range(p):
		key, val = idx_need[i]
		num = [int(char) for char in key.split()]
		num = np.array(num)
		A[i,:] = num
	A = A.astype('int8')
	return A

def gaussianElimination(A):
	Perm = []
	p = A.shape[0]
	for i in range(p):
		row_keep = np.where(np.sum(A, axis=1) > 0)[0]
		A = A[row_keep]
		if i >= A.shape[0]:
			break
		if np.where(A[i:,i] == 1)[0].size == 0:
			col_swap = np.where(A[i,:] == 1)[0][0]
			Perm.append((i,col_swap))
			col_temp = A[:,i] + 0
			A[:,i] = A[:,col_swap] + 0
			A[:,col_swap] = col_temp + 0
		elif A[i,i] == 0:
			row_swap = np.where(A[i:,i] == 1)[0][0] + i
			row_temp = A[i,:] + 0
			A[i,:] = A[row_swap,:] + 0
			A[row_swap,:] = row_temp + 0
		for j in np.where(A[:,i] == 1)[0]:
			if j != i:
				A[j,:] = A[j,:] ^ A[i,:]
	A = A[:p,:]
	return A, Perm

def solutionH(A, Perm):
	G = A[:,A.shape[0]:]
	H = np.append(G,np.eye(G.shape[1]), axis = 0).astype('int8')
	Perm.reverse()
	if Perm != []:
		for pair in Perm:
			i,j = pair
			row_temp = H[i,:] + 0
			H[i,:] = H[j,:] + 0
			H[j,:] = row_temp + 0
	return H

def posInFourier(H):
	posF = np.zeros([2**(H.shape[1]),H.shape[0]])
	for k in range(2**(H.shape[1])):
		a = "{0:b}".format(k)
		a = a.zfill(H.shape[1])
		a = [int(char) for char in str(a)]
		a = np.array(a)
		# print H.dot(a) % 2
		posF[k,:] = H.dot(a) % 2
	return posF

def coefficients(posF, train):
	coeff = np.zeros(posF.shape[0])
	for i in range(posF.shape[0]):
		pos = posF[i,:]
		pos_x = np.where(pos==1)
		e = train[:,pos_x][:,0,:] - 1
		mul = np.sum(e,axis=1) % 2
		mul = -2 * mul + 1
		coeff[i] = np.mean(mul) * 1.0 / (2**posF.shape[1])
	return coeff

