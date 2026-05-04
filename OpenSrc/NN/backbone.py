import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.utils import variational_estimator

from .modules.Time import TimeDomain
from .modules.attention import FusionModule
from .modules.bayesEmbedding import BayesianEmbedding
from .modules.feature_extractor import Frequency_CNN


@variational_estimator
class BayesianCNNEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        time_config = config.get("time", {})
        stft_config = config.get("stft", {})
        freq_config = config.get("frequency", {})
        fusion_config = config.get("fusion", {})
        embedding_config = config.get("embedding", {})

        time_filters = time_config.get("num_filters", 32)
        time_dim = time_filters * 4
        freq_channels = (
            freq_config.get("channels_1", 16),
            freq_config.get("channels_2", 32),
            freq_config.get("channels_3", 64),
        )
        freq_groups = (
            freq_config.get("groups_1", 4),
            freq_config.get("groups_2", 8),
            freq_config.get("groups_3", 8),
        )
        freq_dim = freq_channels[-1]

        self.stft_config = {
            "n_fft": stft_config.get("n_fft", 64),
            "hop_length": stft_config.get("hop_length", 16),
            "win_length": stft_config.get("win_length", 64),
            "spec_height": stft_config.get("spec_height", 128),
            "spec_width": stft_config.get("spec_width", 128),
            "log_eps": stft_config.get("log_eps", 1e-8),
        }

        self.time_encoder = TimeDomain(
            c_in=time_config.get("input_channels", 1),
            nf=time_filters,
            is_gap=time_config.get("use_global_pool", False),
            depth=time_config.get("depth", 3),
            kernel_size=time_config.get("kernel_size", 15),
            residual=time_config.get("residual", True),
            bottleneck=time_config.get("bottleneck", True),
        )
        self.freq_encoder = Frequency_CNN(
            input_channels=freq_config.get("input_channels", 1),
            channels=freq_channels,
            norm_groups=freq_groups,
        )
        self.fusion = FusionModule(
            channel_attn=time_dim,
            fusion_dim=time_dim,
            freq_dim=freq_dim,
            channel_reduction=fusion_config.get("channel_reduction", 8),
            min_channel_hidden=fusion_config.get("min_channel_hidden", 4),
            temporal_kernel_size=fusion_config.get("temporal_kernel_size", 7),
            hidden_dim=fusion_config.get("hidden_dim", 8),
            beta_init=fusion_config.get("beta_init", 0.1),
        )
        self.embedding = BayesianEmbedding(
            in_dim=time_dim,
            latent_dim=embedding_config.get("latent_dim", 128),
            embed_dim=embedding_config.get("embed_dim", 128),
        )

    def stft(self, x):
        window = torch.hann_window(self.stft_config["win_length"]).to(x.device)
        stft = torch.stft(
            x,
            n_fft=self.stft_config["n_fft"],
            hop_length=self.stft_config["hop_length"],
            win_length=self.stft_config["win_length"],
            window=window,
            return_complex=True,
        )
        spec = torch.abs(stft)
        spec = torch.log(spec + self.stft_config["log_eps"])
        spec = spec.unsqueeze(1)
        spec = F.interpolate(
            spec,
            size=(self.stft_config["spec_height"], self.stft_config["spec_width"]),
            mode="bilinear",
            align_corners=False,
        )
        return spec

    def forward(self, x):
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

        freq = self.stft(x)
        freq = self.freq_encoder(freq)
        freq = freq.unsqueeze(1)

        time = self.time_encoder(x)
        feature = self.fusion(time, freq)
        feature = F.adaptive_avg_pool1d(feature, 1).squeeze(-1)

        feature, mu, logvar = self.embedding(feature)
        return feature, mu, logvar
