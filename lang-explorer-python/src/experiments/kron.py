#!/usr/bin/env python

import numpy as np

def find_kron_val(k, i, j, matrix):
	if k == 0:
		return matrix[int(i), int(j)]

	rc = 2**(k+1)
	rc2 = rc / 2

	if i >= rc2 and j >= rc2:
		return matrix[1, 1] * find_kron_val(k - 1, i - rc2, j - rc2, matrix)
	elif i < rc2 and j < rc2:
		return matrix[0, 0] * find_kron_val(k - 1, i, j, matrix)
	elif i < rc2 and j >= rc2:
		return matrix[0, 1] * find_kron_val(k - 1, i, j - rc2, matrix)
	elif i >= rc2 and j < rc2:
		return matrix[1, 0] * find_kron_val(k - 1, i - rc2, j, matrix)

def main():
	kron = np.array([[0.1, 0.2], [0.3, 0.4]])

	canonical = kron
	for i in range(2):
		canonical = np.kron(canonical, kron)

	new = np.empty((8, 8))

	for i in range(8):
		for j in range(8):
			new[i, j] = find_kron_val(2, i, j, kron)
	
	print(kron)
	print()
	print(canonical)
	print()
	print(new)

	# for i in range(8):
	# 	for j in range(8):
	# 		if new[i, j] != canonical[i, j]:
	# 			print(f"error! {i} {j} {new[i, j]} {canonical[i, j]}")
	
if __name__ == "__main__":
	main()
