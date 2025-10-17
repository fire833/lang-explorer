
import json
import sys
import matplotlib.pyplot as plt
import numpy as np

def compute_bucket_borders(centers):
    if len(centers) < 2:
        return centers
    
    widths = np.diff(centers)
    avg_width = np.mean(widths)
    
    borders = []
    
    # First border: extend left by half width
    borders.append(centers[0] - avg_width / 2)
    
    # Middle borders: midpoints between consecutive centers
    for i in range(len(centers) - 1):
        borders.append((centers[i] + centers[i + 1]) / 2)
    
    # Last border: extend right by half width
    borders.append(centers[-1] + avg_width / 2)
    
    return borders


def plot_histogram(name, directory, histogram_data, title_prefix=""):
    centers = [item[0] for item in histogram_data]
    counts = [item[1] for item in histogram_data]
    
    borders = compute_bucket_borders(centers)
    
    plt.figure(figsize=(10, 6))
    plt.bar(centers, counts, width=np.diff(borders), align="center", edgecolor="black", alpha=0.7)
    
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"{name} {title_prefix} Similarity Distribution Histogram")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{directory}/{name.replace(":", "_").replace(" ", "_")}_histogram.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def generate_moments_table(ast_moments, embedding_distributions):
    """
    Generate a LaTeX table for distribution moments.
    """
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|c|}")
    print("\\hline")
    print("Distribution & Mean & Variance & Skewness & Kurtosis \\\\")
    print("\\hline")
    
    # AST distribution
    moments = ast_moments
    print(f"AST & {moments[0]:.6f} & {moments[1]:.6f} & {moments[2]:.6f} & {moments[3]:.6f} \\\\")
    print("\\hline")
    
    # Embedding distributions
    for dist in embedding_distributions:
        name = dist["name"]
        moments = dist["moments"]
        print(f"{name} & {moments[0]:.6f} & {moments[1]:.6f} & {moments[2]:.6f} & {moments[3]:.6f} \\\\")
        print("\\hline")
    
    print("\\end{tabular}")
    print("\\caption{Statistical moments of distributions}")
    print("\\label{tab:moments}")
    print("\\end{table}")


def generate_similarity_table(similarity_results, embedding_distributions):
    """
    Generate a LaTeX table for similarity results.
    """
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|c|c|}")
    print("\\hline")
    print("Distribution & Simple Average & Weighted Average & Chi-Squared Average & Normalized Simple Average & Normalized Chi-Squared Average \\\\")
    print("\\hline")
    
    for i, dist in enumerate(embedding_distributions):
        name = dist["name"]
        results = similarity_results[i]
        print(f"{name} & {results[0]:.6f} & {results[1]:.6f} & {results[2]:.6f} & {results[3]:.6f} & {results[4]:.6f} \\\\")
        print("\\hline")
    
    print("\\end{tabular}")
    print("\\caption{Similarity results for embedding distributions compared to AST distribution}")
    print("\\label{tab:similarity}")
    print("\\end{table}")

def process_experiment(args):
	f = open(args.input, "r")
	data = json.load(f)
	
	ast_data = data["ast_distribution"]
	plot_histogram("AST", args.output, ast_data["histogram"], "")

	for embedding in data["embedding_distributions"]:
		if args.plots:
			plot_histogram(embedding["name"], args.output, embedding["histogram"], "Embedding")

	if args.tables:
	    generate_moments_table(data["ast_distribution"]["moments"], data["embedding_distributions"])
	    generate_similarity_table(data["similarity_results"], data["embedding_distributions"])
