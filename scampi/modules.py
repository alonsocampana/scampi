import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import init_weights
import os

try:
    NO_ATTN = bool(os.environ["SCAMPI_NO_ATTN"])
    NO_TRANS = bool(os.environ["SCAMPI_NO_TRANS"])
except KeyError:
    NO_ATTN = False
    NO_TRANS = False


class TransformerEncoderGated(nn.TransformerEncoder):
    """
    Set of transformer layers with learnable gates.

    Args:
    encoder_layer: A transformer encoder layer
    num_layers: Number of stacked encoder layers
    """

    def __init__(self,
                 encoder_layer: torch.nn.Module,
                 num_layers: int, **kwargs):
        super().__init__(encoder_layer, num_layers, **kwargs)
        self.gates = nn.Parameter(torch.Tensor(num_layers))
        self.layers = nn.ModuleList([encoder_layer for i in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        range_gates = torch.sigmoid(self.gates)
        for i, layer in enumerate(self.layers):
            src = (range_gates[i])*layer(src) + (1-range_gates[i])*src
        return src


class TransformerProt(nn.Module):
    """
    Protein transformer model used for pretraining

    """

    def __init__(self):
        super().__init__()
        self.do = nn.Dropout2d(p=0.05)
        tl = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dropout=0.2, batch_first=True)
        self.transformer = TransformerEncoderGated(tl, 4)
        self.fc = nn.Sequential(nn.Linear(128, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 128),
                                nn.ReLU(),
                                nn.Linear(128, 26))
        self.transformer.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        # x = self.do(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x


class ResBlock(nn.Module):
    """
    Two fully connected layers, with a ReLU and learnable residual connections.

    Args:
    input_size: Number of features for the input and output of the block.
    hidden_space: Width of the hidden layer
    dropout: Probability of dropping a feautre before entering the layer
    """

    def __init__(self,
                 input_size: int,
                 hidden_dim: int,
                 p_dropout: float = 0.4):
        super().__init__()
        self.nw = nn.Sequential(nn.Dropout(p_dropout),
                               nn.Linear(input_size, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, input_size))

    def forward(self, x):
        x = self.nw(x)
        return x


class N_ResBlocks(nn.Module):
    """
    A stack of N blocks connected with residual connections and learnable gates

    Args:
    input_dim: Number of features for the input and output of the block.
    hidden_space: Width of the hidden layer
    n_blocks: Number of stacked residual blocks
    p_dropout: Probability of dropping a feautre before entering the layer
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_blocks: int,
                 p_dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([ResBlock(input_dim,
                                              hidden_dim,
                                              p_dropout) for i in range(n_blocks)])
        self.gates = nn.Parameter(torch.Tensor(n_blocks))

    def forward(self, x):
        range_gates = torch.sigmoid(self.gates)
        for i, layer in enumerate(self.layers):
            x = (range_gates[i])*layer(x) + (1-range_gates[i])*x
        return x


class MetEncoder(nn.Module):
    """
    Encoder of the metabolite features.

    Args:
    init_dim: Initial number of features
    target_dim: Encoded dimension
    """

    def __init__(self,
                 init_dim: int,
                 target_dim: int):
        super().__init__()
        self.nw = nn.Sequential(nn.Linear(init_dim, target_dim*32),
                               nn.ReLU(),
                               nn.Linear(target_dim*32, target_dim*2),
                               nn.ReLU(),
                               nn.BatchNorm1d(target_dim*2),
                               nn.Linear(target_dim*2, target_dim))
    def forward(self, x):
        return self.nw(x)


class AttnBlock(nn.Module):
    """
    Cross-attention between the protein sequence and the metabolite features.

    Args:
    input_dim: Number of features of the input
    hidden_dim: Number of hidden features for mapping to attention space
    output_dim: Number of features output
    p_dropout: Dropout probability while mapping to attention space
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 p_dropout: float):
        super().__init__()
        self.fc_prot = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(p = p_dropout),
                                   nn.Linear(hidden_dim, input_dim))
        self.fc_met = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(p = p_dropout),
                                   nn.Linear(hidden_dim, input_dim))
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        x = self.fc_met(x)
        y = self.fc_prot(y)
        z = F.softmax(torch.bmm(y, x.unsqueeze(2))/np.sqrt(x.shape[1]), dim=1)
        x = torch.bmm(z.transpose(2, 1), y).squeeze()
        # del y, z
        # gc.collect()
        return self.lin(x)


class N_AttnBlock(nn.Module):
    """
    N parallel cross-attention heads.

    Args:
    input_dim: Number of features of the input
    hidden_dim: Number of hidden features for mapping to attention space
    output_dim: Number of features output
    n_blocks: Number of parallel attention heads
    p_dropout: Dropout probability while mapping to attention space
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_blocks: int,
                 p_dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([AttnBlock(input_dim,
                                               hidden_dim,
                                               output_dim,
                                               p_dropout) for i in range(n_blocks)])

    def forward(self, x, y):
        y_outs = []
        for block in self.blocks:
            temp_y = block(x, y)
            y_outs.append(temp_y)
        # del temp_y, x
        # gc.collect()
        return torch.cat(y_outs, axis=1)


class IntegrativeModel(nn.Module):
    """
    Model mapping a metabolite (as a series of fingerprint-derived
    features) and a protein (as a sequence) to a score representing their
    interaction probability.

    Args:
    hidden_dim:
    init_dim_met: Initial number of metabolite features
    dim_prot: Number of features for each sequence element
    n_attn_blocks: Number of attention block (heads)
    n_res_blocks: Number of attention block (heads)
    p_dropout: Dropout probability during several steps of the model
    """

    def __init__(self,
                 hidden_dim: int,
                 init_dim_met: int,
                 dim_prot: int,
                 n_attn_blocks: int,
                 n_res_blocks: int,
                 p_dropout: float):
        super().__init__()
        tl = nn.TransformerEncoderLayer(d_model=dim_prot,
                                        nhead=8,
                                        dropout=p_dropout,
                                        batch_first=True)
        self.prot_transformer = TransformerEncoderGated(tl, 4)
        self.met_encoder = MetEncoder(init_dim_met, dim_prot)
        assert dim_prot % n_attn_blocks == 0
        if NO_ATTN:
            pass
        else:
            self.attn = N_AttnBlock(dim_prot,
                                    hidden_dim,
                                    dim_prot//n_attn_blocks,
                                    n_attn_blocks,
                                    p_dropout/10)
            self.attn.apply(init_weights)
        dim_res = dim_prot * 2
        self.integrative = nn.Sequential(N_ResBlocks(dim_res,
                                                     dim_res * 4,
                                                     n_res_blocks,
                                                     p_dropout))
        self.classifier = nn.Linear(dim_res, 1)
        self.met_encoder.apply(init_weights)
        self.prot_transformer.apply(init_weights)
        self.integrative.apply(init_weights)
        self.classifier.apply(init_weights)

    def set_cold(self):
        """
        Turns off gradient for the protein transformer.
        """
        for p in self.prot_transformer.parameters():
            p.requires_grad = False

    def set_warm(self):
        """
        Turns on gradient for the protein transformer.
        """
        for p in self.prot_transformer.parameters():
            p.requires_grad = True

    def forward(self, x, y):
        x = self.met_encoder(x)
        if NO_TRANS:
            pass
        else:
            y = self.prot_transformer(y)
        if NO_ATTN:
            x = self.integrative(torch.cat([x, y.mean(axis=1)], axis=1))
        else:
            y = self.attn(x, y)
            x = self.integrative(torch.cat([x, y], axis=1))
        x = self.classifier(x)
        return x
