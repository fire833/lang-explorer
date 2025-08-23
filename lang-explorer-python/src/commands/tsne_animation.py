
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE

def animate(args):
	files = sorted(glob.glob(args.input))  # adjust pattern
	snapshots = []

	for f in files:
	    print(f"Processing {f}...")
	    data = pd.read_csv(f)  # rows = samples, cols = features (+ maybe label)
	
	    tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
	    embedding = tsne.fit_transform(data.values)
	
	    snapshots.append(embedding)

	fig, ax = plt.subplots(figsize=(len(files), 1))
	sc = ax.scatter([], [], s=10, cmap="tab10")

	def init():
	    sc.set_offsets([])
	    return sc,

	def update(frame):
	    coords, labels = snapshots[frame]
	    sc.set_offsets(coords)
	    if labels is not None:
	        sc.set_array(labels)
	    ax.set_title(f"t-SNE Frame {frame+1}/{len(snapshots)}")
	    return sc,

	ani = FuncAnimation(fig, update, frames=len(snapshots),
	                    init_func=init, blit=True, interval=1000)

	ani.save(args.output, writer="ffmpeg", dpi=150)
