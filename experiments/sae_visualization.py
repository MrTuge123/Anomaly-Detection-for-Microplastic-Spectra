"""Plot held-out SAE activations and spectral features from saved NumPy runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Fixed categorical order (colorblind-checked); cluster i always gets slot i.
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _spectral_plot(ax, wavenumbers, values, **kwargs):
    # Do not join lines across spectral regions absent from the input dataset.
    gaps = np.flatnonzero(np.diff(wavenumbers) > 2 * np.median(np.diff(wavenumbers))) + 1
    ax.plot(np.insert(wavenumbers, gaps, np.nan), np.insert(values, gaps, np.nan), **kwargs)


def _top_neurons(z, threshold, k):
    """Rank neurons by activation frequency; break ties by mean activation."""
    frequency = (z > threshold).mean(axis=0)
    return [int(i) for i in np.lexsort((-z.mean(axis=0), -frequency))[:k]]


def plot_sae_run(path, neuron=None, threshold=1e-6, top_k=5):
    """Show activations, frequency, and decoder features with top examples.

    ``neuron`` may be an int or a list of ints. ``None`` selects the ``top_k``
    most frequently active neurons on the validation spectra (ties broken by
    mean activation), most frequent first. Frequencies and example selection
    use ``activation > threshold``. Decoder curves are unit-norm features in
    the model's scaled input space.
    """
    run = _load(path)
    z, features = run["activations"], run["decoder_features"]
    polymer = str(run["polymer"].item())
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    if neuron is None:
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        neurons = _top_neurons(z, threshold, top_k)
    else:
        neurons = [neuron] if isinstance(neuron, (int, np.integer)) else list(neuron)
    if not neurons or any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 0 <= n < z.shape[1]
        for n in neurons
    ):
        raise ValueError(f"neuron must be an integer or list of integers from 0 to {z.shape[1] - 1}")
    neurons = [int(n) for n in neurons]
    frequency = (z > threshold).mean(axis=0)

    fig, axes = plt.subplots(1 + len(neurons), 2, figsize=(13, 3.6 * (1 + len(neurons))),
                             squeeze=False, layout="constrained")
    heat, frequency_ax = axes[0]
    image = heat.imshow(z, aspect="auto", interpolation="nearest", cmap="magma", vmin=0)
    fig.colorbar(image, ax=heat, label="Activation (scaled input units)")
    heat.set(xlabel="Neuron ID", ylabel="Validation spectrum (row order)", title="Feature activations")
    shown = np.isin(np.arange(z.shape[1]), neurons)
    frequency_ax.bar(np.arange(z.shape[1]), frequency * 100,
                     color=np.where(shown, "tab:red", "tab:blue"))
    frequency_ax.set(xlabel="Neuron ID", ylabel="Validation spectra active (%)", ylim=(0, 105),
                     title=f"Activation frequency (threshold {threshold:g}); red = plotted below")

    for n, (feature, examples) in zip(neurons, axes[1:]):
        _spectral_plot(feature, run["wavenumbers"], features[:, n], color="tab:purple")
        feature.axhline(0, color="gray", linewidth=0.7)
        feature.set(xlabel="Wavenumber (cm⁻¹)", ylabel="Unit-norm decoder weight",
                    title=f"Neuron {n} (active in {frequency[n]:.0%}): pattern vs. training mean")

        active_rows = np.flatnonzero(z[:, n] > threshold)
        top = active_rows[np.argsort(-z[active_rows, n])[:5]]
        for row in top:
            _spectral_plot(examples, run["wavenumbers"], run["spectra"][row], alpha=0.8,
                           label=f"ID {run['sample_ids'][row]}: z={z[row, n]:.3g}")
        if len(top):
            examples.legend(fontsize=8)
        else:
            examples.text(0.5, 0.5, "No validation activations above threshold",
                          ha="center", va="center", transform=examples.transAxes)
        examples.set(xlabel="Wavenumber (cm⁻¹)", ylabel="Measured intensity",
                     title=f"Neuron {n}: up to 5 strongest real examples")
    fig.suptitle(f"{polymer} — held-out SAE features")
    return fig


def compare_sae_runs(directory, threshold=1e-6):
    """Compare independent polymer models; neuron IDs are local to each model.

    Heatmaps share a color limit but each model uses its own training-derived
    input scale. Compare sparsity and within-model structure; raw activation
    magnitudes are not calibrated chemical quantities across polymers.
    """
    runs = [_load(path) for path in sorted(Path(directory).glob("*.npz"))]
    if not runs:
        raise ValueError(f"No saved SAE runs in {directory}")
    vmax = max(float(run["activations"].max()) for run in runs) or 1.0
    fig, axes = plt.subplots(len(runs), 2, figsize=(13, 3.2 * len(runs)),
                             squeeze=False, layout="constrained")
    for run, (heat, frequency) in zip(runs, axes):
        z = run["activations"]
        name = str(run["polymer"].item())
        image = heat.imshow(z, aspect="auto", interpolation="nearest", cmap="magma",
                            vmin=0, vmax=vmax)
        heat.set(title=f"{name}: {len(z)} validation spectra", xlabel="Local neuron ID",
                 ylabel="Validation row")
        frequency.bar(np.arange(z.shape[1]), (z > threshold).mean(axis=0) * 100)
        frequency.set(xlabel="Local neuron ID", ylabel="Spectra active (%)", ylim=(0, 105),
                      title=f"Mean active count: {(z > threshold).sum(axis=1).mean():.1f}/{z.shape[1]}")
    fig.colorbar(image, ax=axes[:, 0].tolist(), label="Activation (per-model scaled units)")
    fig.suptitle("Independent polymer SAEs — neuron IDs are not aligned between models")
    return fig


def cluster_activations(z, n_clusters=None, max_clusters=6, seed=0):
    """K-means on raw activation vectors (rows of ``z``).

    ``n_clusters=None`` tries k = 2..max_clusters and keeps the k with the best
    silhouette score. Labels are renumbered so cluster 0 is the largest.
    Returns ``(labels, k, silhouette_by_k)``.
    """
    n = len(z)
    if n < 3:
        raise ValueError("Need at least 3 spectra to cluster")
    if not 2 <= max_clusters <= len(CLUSTER_COLORS):
        raise ValueError(f"max_clusters must be from 2 to {len(CLUSTER_COLORS)}")
    candidates = range(2, min(max_clusters, n - 1) + 1) if n_clusters is None else [n_clusters]
    if n_clusters is not None and not 2 <= n_clusters <= min(len(CLUSTER_COLORS), n - 1):
        raise ValueError(f"n_clusters must be from 2 to {min(len(CLUSTER_COLORS), n - 1)}")
    fits, scores = {}, {}
    for k in candidates:
        fits[k] = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(z)
        distinct = len(np.unique(fits[k]))
        scores[k] = silhouette_score(z, fits[k]) if 2 <= distinct < n else float("nan")
    k = max(scores, key=lambda key: -np.inf if np.isnan(scores[key]) else scores[key])
    sizes = np.bincount(fits[k], minlength=k)
    relabel = np.empty(k, dtype=int)
    relabel[np.argsort(-sizes, kind="stable")] = np.arange(k)
    return relabel[fits[k]], k, scores


def plot_sae_clusters(path, n_clusters=None, max_clusters=6, seed=0):
    """Cluster held-out spectra by activation strength and overlay each cluster.

    Top row: activation heatmap sorted by cluster, and a 2-D PCA view of the
    activation vectors. Then one row per cluster: its mean activation per
    neuron (which features define it) and its measured spectra overlaid with
    the cluster mean. Returns ``(fig, labels)``; labels are 0-based.
    """
    run = _load(path)
    z, spectra, wavenumbers = run["activations"], run["spectra"], run["wavenumbers"]
    polymer = str(run["polymer"].item())
    labels, k, scores = cluster_activations(z, n_clusters, max_clusters, seed)
    colors = CLUSTER_COLORS[:k]
    sizes = np.bincount(labels, minlength=k)

    fig, axes = plt.subplots(1 + k, 2, figsize=(13, 3.6 * (1 + k)),
                             squeeze=False, layout="constrained")
    heat, scatter = axes[0]

    order = np.argsort(labels, kind="stable")
    image = heat.imshow(z[order], aspect="auto", interpolation="nearest", cmap="magma", vmin=0)
    fig.colorbar(image, ax=heat, label="Activation (scaled input units)")
    bounds = np.cumsum(sizes)[:-1] - 0.5
    for y in bounds:
        heat.axhline(y, color="white", linewidth=1.5)
    centers = np.cumsum(sizes) - sizes / 2 - 0.5
    heat.set_yticks(centers, [f"C{c + 1} (n={sizes[c]})" for c in range(k)])
    heat.set(xlabel="Neuron ID", title="Activations sorted by cluster")

    points = PCA(n_components=2).fit(z)
    xy = points.transform(z)
    for c in range(k):
        scatter.scatter(*xy[labels == c].T, s=48, color=colors[c], edgecolor="white",
                        linewidth=1, label=f"Cluster {c + 1} (n={sizes[c]})")
    ratio = points.explained_variance_ratio_ * 100
    how = "chosen by silhouette" if n_clusters is None else "set manually"
    scatter.set(xlabel=f"PC1 of activations ({ratio[0]:.0f}%)",
                ylabel=f"PC2 of activations ({ratio[1]:.0f}%)",
                title=f"k = {k} ({how}; silhouette {scores[k]:.2f})")
    scatter.legend(fontsize=8)

    mean_z = np.stack([z[labels == c].mean(axis=0) for c in range(k)])
    z_max = mean_z.max() * 1.1 or 1.0
    y_lo, y_hi = spectra.min(), spectra.max()
    pad = 0.05 * (y_hi - y_lo)
    overall = spectra.mean(axis=0)
    for c, (bars, overlay) in enumerate(axes[1:]):
        bars.bar(np.arange(z.shape[1]), mean_z[c], color=colors[c])
        top3 = np.argsort(-mean_z[c])[:3]
        bars.set(xlabel="Neuron ID", ylabel="Mean activation", ylim=(0, z_max),
                 title=f"Cluster {c + 1}: defining neurons {', '.join(map(str, top3))}")

        members = spectra[labels == c]
        for i, spectrum in enumerate(members):
            _spectral_plot(overlay, wavenumbers, spectrum, color=colors[c], alpha=0.35,
                           linewidth=1, label="Member spectra" if i == 0 else None)
        _spectral_plot(overlay, wavenumbers, overall, color="gray", linestyle="--",
                       linewidth=1.2, label="All-validation mean")
        _spectral_plot(overlay, wavenumbers, members.mean(axis=0), color="black",
                       linewidth=2, label="Cluster mean")
        overlay.set(xlabel="Wavenumber (cm⁻¹)", ylabel="Measured intensity",
                    ylim=(y_lo - pad, y_hi + pad),
                    title=f"Cluster {c + 1}: {len(members)} spectra overlaid")
        overlay.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{polymer} — held-out spectra clustered by SAE activations")
    return fig, labels
