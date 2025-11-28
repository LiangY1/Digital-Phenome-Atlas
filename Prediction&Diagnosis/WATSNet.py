import torch
import torch.nn as nn
from tsai.all import *

class pheNN(nn.Module):
    def __init__(self, input_size, output_size, depth, width):
        super(pheNN, self).__init__()
        self.inlayer = nn.Linear(input_size, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.outlayer = nn.Linear(width, output_size) 
        
    def forward(self, x):   
        out_main = self.inlayer(x)
        for layer in self.layers:
            out_main = nn.ReLU()(layer(out_main))
        out_main = self.outlayer(out_main)

        return out_main

class WeightedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.65))  

    def forward(self, x1, x2):
        return self.alpha * x1 + (1 - self.alpha) * x2

class CNNWithEmbeddingADPCov(nn.Module):
    def __init__(self, base_model, embedding_dim=3, num_days=7,num_classes=2,ADP_dim=None,cov_dim=None,depth = None,width = None):
        super(CNNWithEmbeddingADPCov, self).__init__()
        
        self.base_model = base_model # ResNetPlus from TSAI
        
        self.embedding = nn.Embedding(num_embeddings=num_days + 1, embedding_dim=embedding_dim)

        self.FCNN_main = pheNN(input_size=cov_dim+1, output_size=num_classes, depth=2, width=50)
        self.FCNN_shortcut = pheNN(input_size=cov_dim, output_size=num_classes, depth=2, width=50)
        self.FCNN_ADP = pheNN(input_size=ADP_dim, output_size=num_classes, depth=depth, width=width)
        self.WeightedFusion = WeightedFusion()

    def forward(self, x_combined):
        x,ADP,cov = x_combined
        x_embedded = self.embedding(x[:,1,:].long()) 
        x_embedded = x_embedded.transpose(1,2)
        out = self.base_model(torch.cat([x[:,0,:].unsqueeze(1),x_embedded],dim=1)) 

        result_ADP = self.FCNN_ADP(ADP)
        fused_result = self.WeightedFusion(out, result_ADP)

        result_main = self.FCNN_main(torch.cat([cov,fused_result],dim = 1))
        result_shortcut = self.FCNN_shortcut(cov)
        return result_main+result_shortcut
        
    def freeze_all_except_shortcut(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.FCNN_main.parameters():
            param.requires_grad = False
        for param in self.FCNN_ADP.parameters():
            param.requires_grad = False
        for param in self.WeightedFusion.parameters():
            param.requires_grad = False
        for param in self.FCNN_shortcut.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
