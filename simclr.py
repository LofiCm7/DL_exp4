import torch
import torch.nn as nn
import torch.nn.functional as F

from net import build_encoder


def nt_xent_loss(z_i, z_j, temperature=0.5):
    if z_i.ndim != 2 or z_i.shape != z_j.shape:
        raise ValueError("z_i and z_j must have shape (batch_size, feature_dim).")
    if z_i.size(0) < 2:
        raise ValueError("batch_size must be at least 2.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    batch_size = z_i.size(0)
    logits = torch.cat([z_i, z_j], dim=0)
    logits = F.normalize(logits, dim=1) @ F.normalize(logits, dim=1).T
    logits = logits / temperature
    logits.fill_diagonal_(float("-inf"))

    targets = torch.arange(batch_size, device=logits.device)
    targets = torch.cat([targets + batch_size, targets])
    return F.cross_entropy(logits, targets)


def info_nce_loss(z_i, z_j, temperature=0.5):
    return nt_xent_loss(z_i, z_j, temperature)


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        return nt_xent_loss(z_i, z_j, self.temperature)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=64, hidden_dim=None):
        super().__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        return self.layers(x)


class SimCLRModel(nn.Module):
    def __init__(self, encoder_name="resnet18", projection_dim=64, projection_hidden_dim=None):
        super().__init__()
        self.encoder, self.feature_dim = build_encoder(encoder_name)
        self.projection_head = ProjectionHead(
            self.feature_dim,
            projection_dim,
            projection_hidden_dim,
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections


def load_pretrained_encoder(checkpoint_path, encoder_name=None, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    checkpoint_model = checkpoint["model"]

    if encoder_name is None:
        encoder_name = checkpoint_model
    elif encoder_name != checkpoint_model:
        raise ValueError(
            f"Checkpoint was trained with model '{checkpoint_model}', but got '{encoder_name}'."
        )

    encoder, feature_dim = build_encoder(encoder_name)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    return encoder, feature_dim, encoder_name


class LinearProbeModel(nn.Module):
    def __init__(self, encoder, feature_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)


def load_linear_probe_model(checkpoint_path, num_classes, encoder_name=None, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    checkpoint_model = checkpoint["model"]

    if encoder_name is None:
        encoder_name = checkpoint_model
    elif encoder_name != checkpoint_model:
        raise ValueError(
            f"Checkpoint was trained with model '{checkpoint_model}', but got '{encoder_name}'."
        )

    encoder, feature_dim = build_encoder(encoder_name)
    model = LinearProbeModel(encoder, feature_dim, num_classes)
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    return model
