"""L1-regularized sparse autoencoder for complete single-polymer spectra."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):
    """Reconstruct spectra using nonnegative, L1-penalized feature activations.

    Architecture (defaults shown for 535-channel spectra)::

        spectrum (535) - training mean
        -> Linear(535, 128) -> ReLU
        -> Linear(128, 32) -> ReLU        # sparse code, L1 penalty here
        -> Linear decoder (32 -> 535) + training mean

    Inputs are floating-point batches of preprocessed absorbance spectra with
    shape ``(batch_size, input_dim)``. Fit preprocessing and ``mean_spectrum``
    using training spectra only. The mean is a saved buffer, not a parameter.
    There is no Top-K gate: the number of active features varies by spectrum.

    Decoder columns may have either sign. Their effective unit norms prevent
    shrinking activations and enlarging decoder weights to evade the L1 loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        l1_lambda: float = 1e-3,
        mean_spectrum: Tensor | None = None,
        intermediate_dim: int = 128,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("intermediate_dim", intermediate_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(l1_lambda) or l1_lambda < 0:
            raise ValueError("l1_lambda must be finite and nonnegative")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.l1_lambda = float(l1_lambda)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.ReLU(),
        )
        self._decoder_weight = nn.Parameter(torch.empty(input_dim, hidden_dim))
        nn.init.xavier_uniform_(self._decoder_weight)
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)
        self.register_buffer("mean_spectrum", torch.zeros(input_dim))
        if mean_spectrum is not None:
            self.set_mean_spectrum(mean_spectrum)

    @property
    def decoder_weight(self) -> Tensor:
        """Effective dictionary, shape ``(input_dim, hidden_dim)``.

        Each column is one spectral feature, normalized on every forward pass.
        """
        return F.normalize(self._decoder_weight, p=2, dim=0)

    @torch.no_grad()
    def set_mean_spectrum(self, mean_spectrum: Tensor) -> None:
        """Set the fixed training mean, copying rather than sharing its storage."""
        if mean_spectrum.shape != (self.input_dim,):
            raise ValueError(f"mean_spectrum must have shape ({self.input_dim},)")
        if not torch.isfinite(mean_spectrum).all():
            raise ValueError("mean_spectrum must contain only finite values")
        self.mean_spectrum.copy_(mean_spectrum)

    @staticmethod
    def _check_batch(values: Tensor, width: int, name: str) -> None:
        if values.ndim != 2 or values.shape[1] != width or values.shape[0] == 0:
            raise ValueError(f"{name} must have shape (nonempty batch_size, {width})")

    def encode(self, spectra: Tensor) -> Tensor:
        """Return nonnegative sparse activations, shape ``(batch, hidden_dim)``.

        Centers on the training mean, then applies the two-layer ReLU encoder.
        """
        self._check_batch(spectra, self.input_dim, "spectra")
        return self.encoder(spectra - self.mean_spectrum)

    def decode(self, activations: Tensor) -> Tensor:
        """Combine spectral features and restore the fixed training mean."""
        self._check_batch(activations, self.hidden_dim, "activations")
        return F.linear(activations, self.decoder_weight) + self.mean_spectrum

    def forward(self, spectra: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(reconstructed_spectra, activations)``."""
        activations = self.encode(spectra)
        return self.decode(activations), activations

    def loss(
        self, spectra: Tensor, reconstruction: Tensor, activations: Tensor
    ) -> dict[str, Tensor]:
        """Return differentiable total, reconstruction, and sparsity losses.

        Reconstruction is MSE over samples and wavenumbers. Sparsity is the
        sample-average sum of absolute feature activations. Only the total
        includes ``l1_lambda``; the reported sparsity term is unweighted.
        """
        self._check_batch(spectra, self.input_dim, "spectra")
        self._check_batch(activations, self.hidden_dim, "activations")
        if reconstruction.shape != spectra.shape:
            raise ValueError("reconstruction and spectra must have the same shape")
        if activations.shape[0] != spectra.shape[0]:
            raise ValueError("activations and spectra must have the same batch size")
        reconstruction_loss = F.mse_loss(reconstruction, spectra)
        sparsity_loss = activations.abs().sum(dim=1).mean()
        return {
            "loss": reconstruction_loss + self.l1_lambda * sparsity_loss,
            "reconstruction_loss": reconstruction_loss,
            "sparsity_loss": sparsity_loss,
        }
