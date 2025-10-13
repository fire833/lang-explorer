#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def main():
	x = np.linspace(0, 20, 1000)
	k_values = [3, 4, 6, 7, 9]

	plt.figure(figsize=(10, 6))

	for k in k_values:
	    y = chi2.pdf(x, k)
	    plt.plot(x, y, label=f'k = {k}', linewidth=2)

	plt.xlabel('x', fontsize=12)
	plt.ylabel('Probability Density', fontsize=12)
	plt.title('Chi-Square Probability Density Function', fontsize=14)
	plt.legend(fontsize=10)
	plt.grid(True, alpha=0.3)
	plt.xlim(0, 20)
	plt.ylim(0, 0.5)

	plt.tight_layout()
	plt.savefig("images/chisqplot.svg", dpi=300)

if __name__ == "__main__":
	main()
