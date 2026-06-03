import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell: (x_t, h_{t-1}, c_{t-1}) -> (h_t, c_t)."""

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM with step-by-step interface for streaming inference."""

    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1, bias=True):
        super().__init__()
        self.layers = nn.ModuleList()
        ch_in = input_dim
        for _ in range(num_layers):
            self.layers.append(
                ConvLSTMCell(input_dim=ch_in, hidden_dim=hidden_dim,
                             kernel_size=kernel_size, bias=bias)
            )
            ch_in = hidden_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size, spatial_size, device):
        H, W = spatial_size
        hidden = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
            hidden.append((h, c))
        return hidden

    def forward_step(self, x, hidden):
        """Single-step forward. Returns (output, new_hidden)."""
        new_hidden = []
        input_l = x
        for l, cell in enumerate(self.layers):
            h_prev, c_prev = hidden[l]
            h_l, c_l = cell(input_l, (h_prev, c_prev))
            new_hidden.append((h_l, c_l))
            input_l = h_l
        return input_l, new_hidden
