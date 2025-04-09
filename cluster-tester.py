from dtaidistance import dtw
from dtaidistance.clustering import KMeans
#from tslearn.clustering import KMeans

import numpy as np
#from functorch.dim import use_c
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

signals = np.load('signals/signals.npy')
masks = np.load('signals/masks.npy')

v_to_del = {1: 'p', 2: 'qrs', 3: 't'}
wave_type_to_color = {
    "p": "yellow",
    "qrs": "red",
    "t": "green"
}

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

'''
for i in range(len(segments[2])):
    plt.plot(segments[2][i])
plt.show()
'''

# Теперь кластеризация
series = segments[2]

max = 0
for i in range(len(series)):
    if len(series[i]) > max:
        max = len(series[i])

for i in range(len(series)):
    while len(series[i]) < max:
        series[i].append(0.0)

'''
for i in range(len(series)):
    plt.plot(series[i])
plt.show()
'''

series=np.array(series)

'''
from dtaidistance import clustering
# Custom Hierarchical clustering
model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
cluster_idx = model1.fit(series)
# Augment Hierarchical object to keep track of the full tree
model2 = clustering.HierarchicalTree(model1)
cluster_idx = model2.fit(series)
# SciPy linkage clustering
model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {})
cluster_idx = model3.fit(series)

model2.plot()
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
show_ts_label = lambda idx: "ts-" + str(idx)
model2.plot("hierarchy.png", axes=ax, show_ts_label=show_ts_label,
           show_tr_label=True, ts_label_margin=-10,
           ts_left_margin=10, ts_sample_length=1)


img = mpimg.imread('hierarchy.png')
imgplot = plt.imshow(img)
plt.show()
'''

model = KMeans(k=4, max_it=10, max_dba_it=10, dists_options={"window": 40})
cluster_idx, performed_it = model.fit(series, use_parallel=False)

centers  = model.kmeansplusplus_centers(series=series)

for i in range(len(cluster_idx)):
    cluster_idx[i] = list(cluster_idx[i])
print(cluster_idx, performed_it)

fig, axs = plt.subplots(len(cluster_idx), 2)

for i in range(len(cluster_idx)):
    for j in range(len(cluster_idx[i])):
        axs[i, 0].plot(series[j])
    axs[i, 1].plot(centers[i])

plt.show()



