"""
Global Conditioning for LightLab.

Encodes two global scalar conditions into tokens that are appended to
the SDXL text encoder hidden states and passed through cross-attention:

  1. Ambient light change α ∈ [-1, 1]
  2. Tone-mapping strategy flag: 0.0 (separate) or 1.0 (together)

Per Section 3.4 of LightLab:
  "Global controls are encoded into Fourier features [Tancik et al. 2020]
   followed by a multi-layer perceptron (MLP) that projects them to the
   text embedding dimension. The projections are concatenated to the text
   embeddings and then inserted through the text-to-image cross-attention layers."

For SDXL, the text embedding dimension is 2048 (cross_attention_dim).
"""

import math
import torch
import torch.nn as nn


class FourierFeatureEmbedding(nn.Module):
    """
    Random Fourier Feature encoding for scalar inputs (Tancik et al. 2020).

    Transforms a scalar x ∈ [-1, 1] into a 2*num_frequencies-dimensional
    feature vector via:
        γ(x) = [sin(2π·B·x), cos(2π·B·x)]
    where B ∈ R^{1 × num_frequencies} is a fixed random projection.

    Args:
        num_frequencies: Number of frequency components (output dim = 2×this).
        scale:           Standard deviation of the random frequency matrix.
    """

    def __init__(self, num_frequencies: int = 64, scale: float = 1.0):
        super().__init__()
        # Fixed (non-trainable) random frequency matrix
        B = torch.randn(1, num_frequencies) * scale
        self.register_buffer("B", B)
        self.out_dim = 2 * num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Scalar inputs, shape (B,) or (B, 1).

        Returns:
            Fourier features, shape (B, 2 * num_frequencies).
        """
        x = x.view(-1, 1).float()
        proj = 2.0 * math.pi * (x @ self.B)  # (B, num_frequencies)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, 2*num_freq)


class GlobalConditionEmbedder(nn.Module):
    """
    Encodes two global scalar conditions (ambient α and tone-mapping flag)
    into token embeddings of shape (B, 2, output_dim) that are appended to
    the SDXL cross-attention token sequence.

    The output_dim must match SDXL's cross_attention_dim (2048 for SDXL-base).

    Args:
        num_frequencies: Fourier feature frequencies (output = 2×this per scalar).
        hidden_dim:      MLP hidden layer dimension.
        output_dim:      Must match SDXL cross_attention_dim (default 2048).
        scale:           Fourier feature frequency scale.
    """

    def __init__(
        self,
        num_frequencies: int = 64,
        hidden_dim: int = 512,
        output_dim: int = 2048,
        scale: float = 1.0,
    ):
        super().__init__()

        self.ambient_fourier = FourierFeatureEmbedding(num_frequencies, scale)
        self.tonemap_fourier = FourierFeatureEmbedding(num_frequencies, scale)

        fourier_dim = 2 * num_frequencies  # 128 by default

        def make_mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(fourier_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.ambient_mlp = make_mlp()
        self.tonemap_mlp = make_mlp()

        # Initialize output layers to near-zero for stable early training
        for mlp in [self.ambient_mlp, self.tonemap_mlp]:
            last_layer = mlp[-1]
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        ambient_alpha: torch.Tensor,  # (B,) ∈ [-1, 1]
        tonemap_flag: torch.Tensor,   # (B,) ∈ {0.0, 1.0}
    ) -> torch.Tensor:
        """
        Args:
            ambient_alpha: Ambient light change scalars, shape (B,).
            tonemap_flag:  Tone-mapping strategy flags (0=separate, 1=together), shape (B,).

        Returns:
            Global condition tokens, shape (B, 2, output_dim).
            These are appended to the SDXL text embeddings (B, 77, 2048)
            to form (B, 79, 2048) before cross-attention.
        """
        # Fourier encode each scalar
        amb_feats = self.ambient_fourier(ambient_alpha)    # (B, 128)
        tone_feats = self.tonemap_fourier(tonemap_flag)    # (B, 128)

        # Project to text embedding dimension
        amb_token = self.ambient_mlp(amb_feats).unsqueeze(1)    # (B, 1, 2048)
        tone_token = self.tonemap_mlp(tone_feats).unsqueeze(1)  # (B, 1, 2048)

        return torch.cat([amb_token, tone_token], dim=1)  # (B, 2, 2048)


def append_global_conditions(
    text_embeddings: torch.Tensor,     # (B, seq_len, 2048)
    global_embedder: GlobalConditionEmbedder,
    ambient_alpha: torch.Tensor,       # (B,)
    tonemap_flag: torch.Tensor,        # (B,)
) -> torch.Tensor:
    """
    Convenience function: encode global conditions and append to text embeddings.

    Args:
        text_embeddings: SDXL encoder hidden states (B, seq_len, 2048).
        global_embedder: Trained GlobalConditionEmbedder.
        ambient_alpha:   Ambient intensity changes (B,).
        tonemap_flag:    Tone-mapping strategy flags (B,).

    Returns:
        Extended hidden states (B, seq_len + 2, 2048).
    """
    global_tokens = global_embedder(ambient_alpha, tonemap_flag)  # (B, 2, 2048)
    return torch.cat([text_embeddings, global_tokens], dim=1)
