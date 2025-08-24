
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE

def animate(args):
    files = sorted(glob.glob(args.input))  # adjust pattern
    snapshots = []

    init = ""

    for i, f in enumerate(files):
        print(f"Processing {f}...")
        data = pd.read_csv(f)  # rows = samples, cols = features (+ maybe label)

        tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
        embedding = tsne.fit_transform(data.values) 
        if i == 0:
    	    init = embedding    
        snapshots.append(embedding)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(init[:,0], init[:,1], s=15, cmap="tab10")

    # def init():
    #     sc.set_offsets([])
    #     return sc,

    def update(frame):
        sc.set_offsets(snapshots[frame])
        ax.set_title(f"t-SNE Frame {frame+1}/{len(snapshots)}")
        return sc,

    ani = FuncAnimation(fig, update, frames=len(snapshots), blit=True, interval=300)

    ani.save(args.output, writer="ffmpeg", dpi=150)
