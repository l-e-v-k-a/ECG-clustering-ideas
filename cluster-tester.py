from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance.clustering import KMeans

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from sklearn.utils import shuffle

signals = np.load('signals/signals.npy')
masks = np.load('signals/masks.npy')

def extract_segments(array, mask):
    current_segment = []
    current_label = None

    for element, label in zip(array, mask):
        if label != current_label:
            if current_label is not None:
                if current_label in segments:
                    segments[current_label].append(current_segment)
                else:
                    segments[current_label] = [current_segment]
            current_segment = [element]
            current_label = label
        else:
            current_segment.append(element)

    if current_label is not None:
        if current_label in segments:
            segments[current_label].append(current_segment)
        else:
            segments[current_label] = [current_segment]

    return segments

def cut_end_zeros(array):
    plot_series = array
    for _ in range(len(plot_series) - 1, 0, -1):
        if plot_series[_] != 0:
            return plot_series
        else:
            plot_series = np.delete(plot_series, _)


segments = {}
nums = list(range(10))+[29,31,33,41,48,53,114,155,54,153,73]

for i in nums:
    extract_segments(signals[i, 1, :], masks[i, 1, :])

# Теперь кластеризация
series = segments[2]

max = 0
for i in range(len(series)):
    if len(series[i]) > max:
        max = len(series[i])

for i in range(len(series)):
    while len(series[i]) < max:
        series[i].append(0.0)


series=np.array(series)

'''
# Кластеризация через KMeans

model = KMeans(k=3, max_it=15, max_dba_it=15, dists_options={"window": 40})
cluster_idx, performed_it = model.fit(series, use_parallel=False)
centers  = model.kmeansplusplus_centers(series=series)

fig, axs = plt.subplots(len(cluster_idx), 2)

for i in range(len(cluster_idx)):
    cluster_idx[i] = list(cluster_idx[i])
print(cluster_idx, performed_it)

for i in range(len(cluster_idx)):
    for j in range(0, len(cluster_idx[i]), 10):
        axs[i, 0].plot(cut_end_zeros(series[j]))
    axs[i, 1].plot(cut_end_zeros(centers[i]))

plt.show()

'''

# Собираем всех по dtw
potential_centroids  = [0, 101, -15]

s1 = series[potential_centroids[0]]
s2 = series[potential_centroids[1]]
s3 = series[potential_centroids[2]]


plt.plot(cut_end_zeros(s1), c='b')
plt.plot(cut_end_zeros(s2), c='orange')
plt.plot(cut_end_zeros(s3), c='g')
plt.show()


clusters = [[], [], []]
distances_to_centroid = [[], [], []]

for i in range(len(series)):
    if i not in (potential_centroids):
        distances = [dtw.distance(series[i], s1),
                     dtw.distance(series[i], s2),
                     dtw.distance(series[i], s3)]
        cluster_num = distances.index(min(distances))
        clusters[cluster_num].append(series[i])
        distances_to_centroid[cluster_num].append(min(distances))

potential_centroids = [s1,s2,s3]


for i in range(len(clusters)):
    clusters[i], distances_to_centroid[i] = shuffle(clusters[i], distances_to_centroid[i], random_state=1)

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle(f'Cluster number {i + 1}')

    x = 0
    y = 0
    j = 0
    while (y<5):
        if x==0 and y==0:
            axs[0,0].set_title('Centroid')
            axs[0,0].plot(cut_end_zeros(potential_centroids[i]), c = 'orange')
            x+=1
            j += 1
        if j<len(clusters[i]):
            axs[x,y].set_title(f'{distances_to_centroid[i][j]:.3f}')
            axs[x,y].plot(cut_end_zeros(clusters[i][j]), c='black', linewidth = 1)
            j+=1
        x+=1
        if x == 5:
            x = 0
            y+=1
    fig.tight_layout()
    fig.savefig(f'cluster{i+1}.png')
plt.show()