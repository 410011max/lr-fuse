import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SVDLinear(nn.Module):
    def __init__(self, U, S, V, bias=None, sigma_fuse='UV', V_transpose=True) -> None:
        super().__init__()
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)
        
        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
        self.truncation_rank = S.size(0)

        V = V.t() if V_transpose else V

        if sigma_fuse == 'UV':
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == 'U':
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.contiguous()
        elif sigma_fuse == 'V':
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.mul(S.view(-1, 1)).contiguous()
        else:
            raise ValueError("sigma_fuse must be either 'UV', 'U', or 'V'")
    
    def forward(self, inp):
        y = self.BLinear(inp)
        y = self.ALinear(y)
        return y

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        ratio: float,
        ratio_type='rank',
        act_aware=False,
        alpha=1,
        sigma_fuse="UV"
    ):
        if ratio_type == 'param':
            n_params = linear.weight.numel()
            compressed_params = int(n_params * ratio)
            rank = compressed_params // (linear.in_features + linear.out_features)
        elif ratio_type == 'rank':
            rank = int(math.ceil(float(ratio) * 4096))
        else:
            raise ValueError("ratio_type must be either 'param' or 'rank'")
        
        if ratio >= 1:
            print('Full rank 4096, no SVD applied')
            return linear
        
        print(f"SVD Rank: {rank}")

        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = 1  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix *= linear.fisher_info**alpha

            scaling_diag_matrix += 1e-6  # avoid zero division
            w = w * scaling_diag_matrix.view(1, -1)
        Us = []
        Ss = []
        Vs = []
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        # Low rank approximation
        U = U[:, 0:rank]
        S = S[0:rank]
        V = Vh[0:rank, :]
        V.transpose_(0, 1)
        
        if act_aware:
            V = V / scaling_diag_matrix.view(-1, 1)
        Us = [U]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan or inf check
        for S in Ss:
            if (S!=S).any():
                print("nan in S")
                return (
                    nn.Linear(linear.in_features, linear.out_features)
                    .to(linear.weight.dtype)
                    .to(linear.weight.device)
                )
        for U in Us:
            if (U!=U).any():
                print("nan in U")
                return (
                    nn.Linear(linear.in_features, linear.out_features)
                    .to(linear.weight.dtype)
                    .to(linear.weight.device)
                )
        for V in Vs:
            if (V!=V).any():
                print("nan in V")
                return (
                    nn.Linear(linear.in_features, linear.out_features)
                    .to(linear.weight.dtype)
                    .to(linear.weight.device)
                )

        assert len(Us) == len(Ss) == len(Vs) == 1
        new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias,sigma_fuse)
        return new_linear.to(linear.weight.dtype)
    
    @staticmethod
    def from_linear_whiten(
        linear: nn.Linear,
        param_ratio: float,
    ):
        if param_ratio >= 1:
            print(4096)
            return linear
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        
        rank = compressed_params // (linear.in_features + linear.out_features)

        print(f"Rank: {rank}")
        w = linear.weight.data.float()

        try:
            scaling_diag_matrix = linear.scaling_diag_matrix.to(w.device)
        except AttributeError:
            raise FileExistsError("Cache may not be loaded correctly")
        
        # Get the inverse of scaling_diag_matrix
        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))

        # Multiply scaling_diag_matrix to weight matrix
        W_scale = torch.matmul(w, scaling_diag_matrix.to(torch.float32))
        
        U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)
        
        V = torch.matmul(Vt, scaling_matrix_inv)
        
        # Low rank approximation to the target rank
        U = U[:, :rank]
        S = S[:rank]
        V = V[:rank, :]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan or inf check
        if (S!=S).any():
            print("nan in S")
            return (
                nn.Linear(linear.in_features, linear.out_features)
                .to(linear.weight.dtype)
                .to(linear.weight.device)
                )
        if (U!=U).any():
            print("nan in U")
            return (
                nn.Linear(linear.in_features, linear.out_features)
                .to(linear.weight.dtype)
                .to(linear.weight.device)
            )
        if (V!=V).any():
            print("nan in V")
            return (
                nn.Linear(linear.in_features, linear.out_features)
                .to(linear.weight.dtype)
                .to(linear.weight.device)
            )

        new_linear = SVDLinear(U, S, V, bias, V_transpose=False)
        return new_linear.to(linear.weight.dtype)
    