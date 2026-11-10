# Single-polymer sparse autoencoder

`SparseAutoencoder` uses a linear encoder, ReLU activations, and a linear
decoder. The default hidden size is 32. L1 regularization encourages sparsity;
there is no fixed active-neuron count. Effective decoder columns have unit norm
on every forward pass, preventing decoder rescaling from bypassing the penalty.

The objective is `MSE(x, reconstruction) + l1_lambda * mean(sum(abs(z)))`, where
the sum is over features and the mean is over spectra. The mean spectrum is a
fixed buffer saved in the model state. Decoder features can have either sign
and describe deviations from that mean.

## Usage

```python
import torch
from experiments.Models import SparseAutoencoder

# train_spectra: float32 tensor [number_of_spectra, number_of_wavenumbers].
# Use one polymer, a common valid grid, and absorbance rather than raw %T.
# Remove missing/zero-padded values before A = -log10(T / 100).
# Fit any additional preprocessing on the training split only.
model = SparseAutoencoder(
    input_dim=train_spectra.shape[1],
    hidden_dim=32,
    l1_lambda=1e-3,  # Starting value; tune for the chosen input scale.
    mean_spectrum=train_spectra.mean(dim=0),
).to(train_spectra.device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
optimizer.zero_grad()
reconstruction, activations = model(train_spectra)
losses = model.loss(train_spectra, reconstruction, activations)
losses["loss"].backward()
optimizer.step()

model.eval()
with torch.no_grad():
    reconstruction, activations = model(validation_spectra)
    validation_mse = (validation_spectra - reconstruction).square().mean()
    error_per_spectrum = (validation_spectra - reconstruction).square().mean(dim=1)
    active_per_spectrum = (activations > 0).sum(dim=1)
    spectral_features = model.decoder_weight  # [wavenumbers, features]
```

Repeat optimization with training batches and stop using held-out validation
results. Keep related measurements together when splitting data. Compare L1
strengths using reconstruction, active-feature counts, and feature stability;
selecting solely for reconstruction favors zero regularization. The repository
contains only 72 spectra per polymer, so feature assignments are exploratory.
Learned features require independent validation before chemical interpretation.

Save the architecture configuration (`input_dim`, `hidden_dim`, `l1_lambda`)
alongside `model.state_dict()` when checkpointing.

Run the focused checks from the workspace root:

```sh
python -m unittest discover -s experiments/Models/tests -v
```
