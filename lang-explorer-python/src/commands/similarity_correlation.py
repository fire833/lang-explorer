
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr, linregress
import pandas as pd

def similarity_correlation(args):
	# Need to load input as pandas dataframe
	file = f"results/{args.lang}/{args.experiment_number}/normalized_ast_similarity_scores.csv"
	data = pd.read_csv(file)

	print(data.shape)

	print(data.head())

	num_embeddings = data.shape[1] - 2

	# Need to create plots where X axis corresponds to values in column 2, 3, 4, ... and Y axis corresponds to column 1
	ast_scores = data.iloc[20000:21000, 1]

	for i in range(2, num_embeddings + 2):
		embedding_scores = data.iloc[20000:21000, i]
		embedding_system = data.columns[i]

		analysis = correlation_analysis(embedding_scores, ast_scores)

		plt.figure()
		plt.scatter(embedding_scores, ast_scores)
		plt.xlabel(f"Pairwise similarity scores - {embedding_system}")
		plt.ylabel("AST similarity scores")
		plt.title(f"Similarity Correlation for {args.lang} - {embedding_system}")
		plt.savefig(f"images/{args.lang}/similarity_correlation_{embedding_system}.jpeg")
		plt.close()

def correlation_analysis(x, y):
	# Pearson correlation
	pearson_r, pearson_p = pearsonr(x, y, alternative="greater")
    
	# Spearman correlation
	spearman_r, spearman_p = spearmanr(x, y, alternative="greater")

	# Linear regression
	slope, intercept, regression_r, regression_p, std_err = linregress(x, y, alternative="greater")
	rsq = regression_r ** 2

	# Create results dictionary
	results = {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
		"regression_r": regression_r,
		"regression_p": regression_p,
		"regression_rsq": rsq,
    }
    
	return results


