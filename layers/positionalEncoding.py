import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, K: int):
        """
            P_{u,v} = [cos(a_i . d(u,v)); sin(a_i . d(u,v))] for i \in [1, ..., K]
            d(u,v) shape [batch, 1]
            P_{u,v} shape [batch, 2K]
            NOTE: Parameters a_i are learnable
        """
        super(PositionalEncoding, self).__init__()
        self.K = K
        self.A = torch.nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_parameters()

    def forward(self, d:torch.Tensor):
        pos = self.A * d    # shape [batch, K]
        encoding = torch.cat([pos.cos(), pos.sin()], dim=-1) # shape [batch, 2K]
        return encoding
    
    def reset_parameters(self):
        torch.nn.init.uniform_(self.A, a=0, b=1)