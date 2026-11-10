"""Plot held-out SAE activations and spectral features from saved NumPy runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _spectral_plot(ax, wavenumbers, values, **kwargs):
    # Do not join lines across spectral regions absent from the input dataset.
    gaps = np.flatnonzero(np.diff(wavenumbers) > 2 * np.median(np.diff(wavenumbers))) + 1
    ax.plot(np.insert(wavenumbers, gaps, np.nan), np.insert(values, gaps, np.nan), **kwargs)


def plot_sae_run(path, neuron=None, threshold=1e-6):
    """Show activations, frequency, one decoder feature, and its top examples.

    ``neuron=None`` selects the feature with the highest mean validation
    activation. Frequencies and example selection use ``activation > threshold``.
    Decoder curves are unit-norm features in the model's scaled input space.
    """
    run = _load(path)
    z, features = run["activations"], run["decoder_features"]
    polymer = str(run["polymer"].item())
    neuron = int(z.mean(axis=0).argmax()) if neuron is None else neuron
    if not isinstance(neuron, (int, np.integer)) or not 0 <= neuron < z.shape[1]:
        raise ValueError(f"neuron must be an integer from 0 to {z.shape[1] - 1}")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    heat, frequency, feature, examples = axes.ravel()
    image = heat.imshow(z, aspect="auto", interpolation="nearest", cmap="magma", vmin=0)
    fig.colorbar(image, ax=heat, label="Activation (scaled input units)")
    heat.set(xlabel="Neuron ID", ylabel="Validation spectrum (row order)", title="Feature activations")
    frequency.bar(np.arange(z.shape[1]), (z > threshold).mean(axis=0) * 100)
    frequency.set(xlabel="Neuron ID", ylabel="Validation spectra active (%)", ylim=(0, 105),
                  title=f"Activation frequency (threshold {threshold:g})")
    frequency.axvline(neuron, color="tab:red", linestyle="--", alpha=0.7)

    _spectral_plot(feature, run["wavenumbers"], features[:, neuron], color="tab:purple")
    feature.axhline(0, color="gray", linewidth=0.7)
    feature.set(xlabel="Wavenumber (cm⁻¹)", ylabel="Unit-norm decoder weight",
                title=f"Neuron {neuron}: spectral pattern relative to training mean")

    active_rows = np.flatnonzero(z[:, neuron] > threshold)
    top = active_rows[np.argsort(-z[active_rows, neuron])[:5]]
    for row in top:
        _spectral_plot(examples, run["wavenumbers"], run["spectra"][row], alpha=0.8,
                       label=f"ID {run['sample_ids'][row]}: z={z[row, neuron]:.3g}")
    if len(top):
        examples.legend(fontsize=8)
    else:
        examples.text(0.5, 0.5, "No validation activations above threshold",
                      ha="center", va="center", transform=examples.transAxes)
    examples.set(xlabel="Wavenumber (cm⁻¹)", ylabel="Measured intensity",
                 title=f"Neuron {neuron}: up to 5 strongest real examples")
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
