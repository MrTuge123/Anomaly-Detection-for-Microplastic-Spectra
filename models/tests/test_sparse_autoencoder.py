"""Checks for loss scaling, dictionary constraints, and trainability."""

import unittest

import torch

from models import SparseAutoencoder


class SparseAutoencoderTests(unittest.TestCase):
    def test_architecture_shapes(self):
        model = SparseAutoencoder(535)
        linears = [m for m in model.encoder if isinstance(m, torch.nn.Linear)]
        self.assertEqual([(l.in_features, l.out_features) for l in linears],
                         [(535, 128), (128, 32)])
        self.assertEqual(tuple(model.decoder_weight.shape), (535, 32))
        reconstruction, activations = model(torch.randn(4, 535))
        self.assertEqual(tuple(reconstruction.shape), (4, 535))
        self.assertEqual(tuple(activations.shape), (4, 32))
        self.assertTrue((activations >= 0).all())

    def test_l1_is_sample_average_not_feature_average(self):
        model = SparseAutoencoder(3, hidden_dim=2, l1_lambda=0.1)
        spectra = torch.zeros(2, 3)
        activations = torch.tensor([[1.0, 0.0], [2.0, 3.0]])
        losses = model.loss(spectra, spectra, activations)
        torch.testing.assert_close(losses["sparsity_loss"], torch.tensor(3.0))
        torch.testing.assert_close(losses["loss"], torch.tensor(0.3))

    def test_decoder_rescaling_cannot_evade_sparsity_penalty(self):
        torch.manual_seed(4)
        model = SparseAutoencoder(10, hidden_dim=4)
        spectra = torch.randn(5, 10)
        before, activations = model(spectra)
        with torch.no_grad():
            model._decoder_weight.mul_(100)
        after, new_activations = model(spectra)
        torch.testing.assert_close(before, after)
        torch.testing.assert_close(activations, new_activations)
        torch.testing.assert_close(model.decoder_weight.norm(dim=0), torch.ones(4))

    def test_zero_code_restores_mean_and_mean_survives_checkpoint(self):
        mean = torch.linspace(0, 1, 10)
        model = SparseAutoencoder(10, mean_spectrum=mean)
        mean.zero_()  # Constructor must copy its input.
        restored = SparseAutoencoder(10)
        restored.load_state_dict(model.state_dict())
        expected = torch.linspace(0, 1, 10).expand(2, -1)
        torch.testing.assert_close(restored.decode(torch.zeros(2, 32)), expected)
        self.assertFalse(restored.mean_spectrum.requires_grad)

    def test_optimization_reduces_held_out_reconstruction_error(self):
        torch.manual_seed(12)
        grid = torch.linspace(-3, 3, 24)
        bands = torch.stack([torch.exp(-(grid - c).square()) for c in (-1, 1)])
        spectra = torch.rand(48, 2) @ bands
        train, validation = spectra[:32], spectra[32:]
        model = SparseAutoencoder(24, hidden_dim=8, mean_spectrum=train.mean(0))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        with torch.no_grad():
            initial_mse = (model(validation)[0] - validation).square().mean()
        for _ in range(150):
            optimizer.zero_grad()
            reconstruction, activations = model(train)
            model.loss(train, reconstruction, activations)["loss"].backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        with torch.no_grad():
            final_mse = (model(validation)[0] - validation).square().mean()
        self.assertLess(final_mse.item(), initial_mse.item() * 0.5)


if __name__ == "__main__":
    unittest.main()
