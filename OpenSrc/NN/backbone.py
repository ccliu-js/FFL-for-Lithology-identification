# from .modules.bayesEmbedding import  BayesianEmbedding
# from .modules.fusion_attention import LightAttention1D
# from .modules.feature_extractor import TimeCNN, Frequency_CNN

# from modules.bayesEmbedding import  BayesianEmbedding
# from modules.fusion_attention import LightAttention1D
# from modules.feature_extractor import TimeCNN, Frequency_CNN

# from modules.Time import TimeDomain
# from modules.attention import FusionModule


from .modules.bayesEmbedding import  BayesianEmbedding
from .modules.fusion_attention import LightAttention1D
from .modules.feature_extractor import TimeCNN, Frequency_CNN

from .modules.Time import TimeDomain
from .modules.attention import FusionModule
from .modules.attention import LiteSEFusion

import torch.nn as nn
from blitz.utils import variational_estimator
import torch
import torch.nn.functional as F



@variational_estimator
class BayesianCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_encoder = TimeDomain(c_in=1, nf=32)
        
        self.freq_encoder = Frequency_CNN()

        self.fusion = FusionModule(channel_attn=128, fusion_dim=128)
 


        # self.timeattention=nn.ModuleList([
        #     LightAttention1D(32),
        #     LightAttention1D(64),
        #     LightAttention1D(128)
        # ])



        self.embedding = BayesianEmbedding(in_dim=128)

    def stft(self, x):
        window = torch.hann_window(64).to(x.device)
        stft = torch.stft(
            x,
            n_fft=64,
            hop_length=16,
            win_length=64,
            window=window,  
            return_complex=True
        )
        spec = torch.abs(stft)
        spec = torch.log(spec + 1e-8)
        spec = spec.unsqueeze(1)        # [B, 1, 129, 32]
        spec = F.interpolate(
            spec,
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        )
        return spec

    def forward(self, x):#x: [B, 4000]
        # frequency feature extraction  
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        freq=self.stft(x)  # return [B, 1, 128, 128]
        freq=self.freq_encoder(freq)#return  [B, 64]

        # time feature extraction
        time=self.time_encoder(x)# return [f1,f2,f3]  f1:[B, 32, 4000]  f2:[B, 64, 2000]  f3:[B, 128, 1000]

        # # time attention
        # for i in range(3):
        #     time[i]=self.timeattention[i](time[i])
        #     time[i]=F.adaptive_avg_pool1d(time[i],1000)


        # time=torch.cat(time,dim=1)  # [B, 224, 1000]
        # time = F.adaptive_avg_pool1d(time, 1).squeeze(-1)# [B, 224]


        # #cross attention
        # feature=torch.cat([time,freq],dim=1) 

        # time = F.adaptive_avg_pool1d(time, 1).squeeze(-1)  # [B, 128]
        freq=freq.unsqueeze(1)


        feature=self.fusion(time,freq)  
        feature = F.adaptive_avg_pool1d(feature, 1).squeeze(-1)  # [B, 128]

        


        # #bayesian embedding
        feature,mu,logvar=self.embedding(feature)  


        return feature,mu,logvar

        
    


if __name__ == "__main__":
    model = BayesianCNNEncoder()
    x = torch.randn(8, 512)  # batch of 8 samples, each of length 4000
    feature, mu, logvar = model(x)
    print(feature.shape)  # should be [8, 128]