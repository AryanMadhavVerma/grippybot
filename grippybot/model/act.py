"""
ACT (Action Chunking with Transformers) — from scratch.

Architecture:
  Part A: CVAE Encoder (training only) — compresses action sequence to latent z
  Part B: Vision Backbone — ResNet18 → 300 visual tokens
  Part C: Observation Fuser — transformer encoder fuses visual + state + z
  Part D: Action Decoder — transformer decoder generates action chunk
"""

import math
import torch
import torch.nn as nn
import torchvision


def sinusoidal_pos_embedding(n_positions, d_model):
    """1D sinusoidal positional embedding."""
    pe = torch.zeros(n_positions, d_model)
    position = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def sinusoidal_2d_pos_embedding(h, w, d_model):
    """2D sinusoidal positional embedding for spatial feature maps."""
    half_d = d_model // 2
    pe_h = sinusoidal_pos_embedding(h, half_d)  # [h, half_d]
    pe_w = sinusoidal_pos_embedding(w, half_d)  # [w, half_d]
    # Broadcast: each (row, col) gets concat of row_pe and col_pe
    pe_h = pe_h.unsqueeze(1).expand(h, w, half_d)  # [h, w, half_d]
    pe_w = pe_w.unsqueeze(0).expand(h, w, half_d)  # [h, w, half_d]
    pe = torch.cat([pe_h, pe_w], dim=-1)  # [h, w, d_model]
    return pe.reshape(h * w, d_model)  # [h*w, d_model]


class VisionBackbone(nn.Module):
    """ResNet18 → 300 visual tokens with positional embeddings."""

    def __init__(self, d_model=256):
        super().__init__()
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        # Keep everything except avgpool and fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Project 512 channels → d_model
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)
        # 2D positional embeddings for 15x20 grid (480/32=15, 640/32=20)
        self.register_buffer(
            "pos_embed", sinusoidal_2d_pos_embedding(15, 20, d_model)
        )

    def forward(self, image):
        """image: [B, 3, 480, 640] → [B, 300, d_model]"""
        features = self.backbone(image)  # [B, 512, 15, 20]
        features = self.proj(features)   # [B, d_model, 15, 20]
        tokens = features.flatten(2).permute(0, 2, 1)  # [B, 300, d_model]
        tokens = tokens + self.pos_embed  # add positional encoding
        return tokens


class CVAEEncoder(nn.Module):
    """Compresses (state, action_sequence) into latent z. Training only."""

    def __init__(self, d_model=256, latent_dim=32, chunk_size=50, state_dim=5, n_layers=4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(state_dim, d_model)
        self.pos_embed = nn.Embedding(1 + 1 + chunk_size, d_model)  # CLS + state + actions

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
            dropout=0.1, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mu_proj = nn.Linear(d_model, latent_dim)
        self.logvar_proj = nn.Linear(d_model, latent_dim)

    def forward(self, state, actions):
        """
        state: [B, 5], actions: [B, chunk_size, 5]
        Returns: mu [B, latent_dim], logvar [B, latent_dim], z [B, latent_dim]
        """
        B = state.shape[0]
        cls = self.cls_token.expand(B, 1, -1)                    # [B, 1, d]
        state_token = self.state_proj(state).unsqueeze(1)         # [B, 1, d]
        action_tokens = self.action_proj(actions)                 # [B, chunk, d]
        sequence = torch.cat([cls, state_token, action_tokens], dim=1)  # [B, 2+chunk, d]

        # Add positional embeddings
        pos_ids = torch.arange(sequence.shape[1], device=sequence.device)
        sequence = sequence + self.pos_embed(pos_ids)

        encoded = self.transformer(sequence)  # [B, 2+chunk, d]
        cls_out = encoded[:, 0]               # [B, d]

        mu = self.mu_proj(cls_out)            # [B, latent_dim]
        logvar = self.logvar_proj(cls_out)    # [B, latent_dim]

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        return mu, logvar, z


class ObservationFuser(nn.Module):
    """Fuses visual tokens + state + z into enriched representations."""

    def __init__(self, d_model=256, latent_dim=32, state_dim=5, n_layers=4):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, d_model)
        self.z_proj = nn.Linear(latent_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
            dropout=0.1, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, visual_tokens, state, z):
        """
        visual_tokens: [B, 300, d], state: [B, 5], z: [B, latent_dim]
        Returns: [B, 302, d]
        """
        state_token = self.state_proj(state).unsqueeze(1)  # [B, 1, d]
        z_token = self.z_proj(z).unsqueeze(1)              # [B, 1, d]
        fused = torch.cat([visual_tokens, state_token, z_token], dim=1)  # [B, 302, d]
        return self.transformer(fused)


class ActionDecoder(nn.Module):
    """Cross-attends learned queries to fuser output → predicts action chunk."""

    def __init__(self, d_model=256, chunk_size=50, state_dim=5, n_layers=1):
        super().__init__()
        self.query_embed = nn.Embedding(chunk_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
            dropout=0.1, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(d_model, state_dim)

    def forward(self, fuser_output):
        """fuser_output: [B, 302, d] → [B, chunk_size, state_dim]"""
        B = fuser_output.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, chunk, d]
        decoded = self.transformer(tgt=queries, memory=fuser_output)       # [B, chunk, d]
        return self.action_head(decoded)  # [B, chunk, state_dim]


class ACTPolicy(nn.Module):
    """Full ACT policy: vision + CVAE + fuser + decoder."""

    def __init__(self, d_model=256, latent_dim=32, chunk_size=50, state_dim=5,
                 n_encoder_layers=4, n_decoder_layers=1):
        super().__init__()
        self.vision = VisionBackbone(d_model)
        self.cvae = CVAEEncoder(d_model, latent_dim, chunk_size, state_dim, n_encoder_layers)
        self.fuser = ObservationFuser(d_model, latent_dim, state_dim, n_encoder_layers)
        self.decoder = ActionDecoder(d_model, chunk_size, state_dim, n_decoder_layers)
        self.latent_dim = latent_dim

    def forward(self, image, state, actions=None):
        """
        image: [B, 3, 480, 640]
        state: [B, 5] (normalized)
        actions: [B, chunk_size, 5] (normalized, training only)

        Returns: pred_actions [B, chunk_size, 5], mu, logvar
        """
        # Part B: Vision
        visual_tokens = self.vision(image)  # [B, 300, d_model]

        # Part A: CVAE (training) or zero latent (inference)
        if actions is not None and self.training:
            mu, logvar, z = self.cvae(state, actions)
        else:
            B = image.shape[0]
            z = torch.zeros(B, self.latent_dim, device=image.device)
            mu, logvar = None, None

        # Part C: Observation Fuser
        fused = self.fuser(visual_tokens, state, z)  # [B, 302, d_model]

        # Part D: Action Decoder
        pred_actions = self.decoder(fused)  # [B, chunk_size, 5]

        return pred_actions, mu, logvar


if __name__ == "__main__":
    # Shape test
    device = "cpu"
    model = ACTPolicy().to(device)
    image = torch.randn(2, 3, 480, 640, device=device)
    state = torch.randn(2, 5, device=device)
    actions = torch.randn(2, 50, 5, device=device)

    model.train()
    pred, mu, logvar = model(image, state, actions)
    print(f"pred: {pred.shape}")    # [2, 50, 5]
    print(f"mu: {mu.shape}")        # [2, 32]
    print(f"logvar: {logvar.shape}") # [2, 32]

    model.eval()
    pred, mu, logvar = model(image, state)
    print(f"pred (eval): {pred.shape}")  # [2, 50, 5]
    print(f"mu (eval): {mu}")            # None

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
