from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance.clustering import KMeans

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

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

segments = {}
for i in range(5):
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
model = KMeans(k=3, max_it=15, max_dba_it=15, dists_options={"window": 40})
cluster_idx, performed_it = model.fit(series, use_parallel=False)
centers  = model.kmeansplusplus_centers(series=series)

fig, axs = plt.subplots(len(cluster_idx), 2)

for i in range(len(cluster_idx)):
    cluster_idx[i] = list(cluster_idx[i])
print(cluster_idx, performed_it)

for i in range(len(cluster_idx)):
    for j in range(len(cluster_idx[i])):
        axs[i, 0].plot(series[j])
    axs[i, 1].plot(centers[i])

plt.show()
'''


# Собираем всех по dtw
s1 = series[38]
s2 = series[0]
s3 = series[10]

plt.plot(s1, c='b')
plt.plot(s2, c='orange')
plt.plot(s3, c='g')
plt.show()

clusters = {0:[series[38]], 1:[series[0]], 2:[series[10]]}

for i in range(len(series)):
    if i not in (0, 10, 38):
        distances = [dtw.distance(series[i], clusters[0][0]),
                     dtw.distance(series[i], clusters[1][0]),
                     dtw.distance(series[i], clusters[2][0])]
        clusters[distances.index(min(distances))].append(series[i])

fig, axs = plt.subplots(len(clusters), 2)

for i in range(len(clusters)):
    for j in range(len(clusters[i])):
        axs[i, 0].plot(series[j])
    axs[i, 1].plot(clusters[i][0])

plt.show()
