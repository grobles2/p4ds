import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Utility functions, written by A. Guernon (see:  https://github.com/ageron/handson-ml/blob/master/08_dimensionality_reduction.ipynb)


def _plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def _plot_centroids(
    centroids,
    weights=None,
    circle_color='w',
    cross_color='b'
):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(
    clusterer,
    X,
    resolution=1000,
    show_centroids=True,
    show_xlabels=True,
    show_ylabels=True
):
    """
    plots the decision boundaries of the clustering algorithm `clusterer`, 
    plotting also the data points of the training set `X`
    """
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    _plot_data(X)
    if show_centroids:
        _plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


def plot_clusterer_comparison(
    clusterer1,
    clusterer2,
    X,
    title1=None,
    title2=None
):
    """
    Plot the decision boundaries of two clustering algorithms side by side.
    """
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(14, 4.7))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)


def plot_dbscan(
  dbscan,
  X,
  size,
  show_xlabels=True,
  show_ylabels=True
):
    """
    Plot the result of a DBSCAN clusterer
    """
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    
    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)


def plot_silouhette_diagrams(
  X,
  k_values
):
  """
  Plot silouhette diagrams for a K-means clusterer for various values of k
  X: DataFrame/ndarray. The training set
  k_values: a tuple or list of integers. Each integers specifies a number of clusters for which we want
            to plot the silouhette diagram
  """
  plt.figure(figsize=(11, 9))

  k_means_per_k = [
    KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    for k in k_values
  ]

  silhouette_scores = [
    silhouette_score(X, model.labels_) for model in k_means_per_k
  ]

  n_cols = 2
  n_rows = (len(k_values) - 1) // n_cols + 1
  for idx, k in enumerate(k_values):
      plt.subplot(n_rows, n_cols, idx + 1)
      
      y_pred = k_means_per_k[idx].labels_
      silhouette_coefficients = silhouette_samples(X, y_pred)

      padding = len(X) // 30
      pos = padding
      ticks = []
      for i in range(k):
          coeffs = silhouette_coefficients[y_pred == i]
          coeffs.sort()

          color = mpl.cm.Spectral(i / k)
          plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                            facecolor=color, edgecolor=color, alpha=0.7)
          ticks.append(pos + len(coeffs) // 2)
          pos += len(coeffs) + padding

      plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
      plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
      
      if idx % n_cols == 0:
          plt.ylabel("Cluster")

      plt.gca().set_xlim(-0.1, 1)

      if idx >= len(k_values) - n_cols:
          plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
          plt.xlabel("Silhouette Coefficient")
      else:
          plt.tick_params(labelbottom=False)

      plt.axvline(x=silhouette_scores[idx], color="red", linestyle="--")
      plt.title("$k={}$".format(k), fontsize=16)