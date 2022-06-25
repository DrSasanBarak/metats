import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
  """
  A general class for stacked lstm encoder
  """
  def __init__(self, input_size, latent_size, hidden_size, num_layers, directions):
    """
    inputs:
        input_size: dimension of input series
        latent_size: dimension of latent representation
        hidden_size: dimension of lstm hidden cells
        num_layers: number of stacked lstm
        directions: 1 for ordinary and 2 for bi-directional lstm
    """
    super().__init__()

    is_bidirectional = (directions == 2)

    self.lstm = nn.LSTM(input_size, hidden_size=hidden_size,
                        bidirectional=is_bidirectional, num_layers=num_layers,
                        proj_size=latent_size)
    
    self.latent_size = latent_size
    self.num_layers = num_layers
    self.directions = directions
    self.hidden_size = hidden_size
  
  def get_initial(self, batch_size):
    h_size = (self.directions * self.num_layers, batch_size, self.latent_size)
    c_size = (self.directions * self.num_layers, batch_size, self.hidden_size)
    h0 = torch.zeros(h_size)
    c0 = torch.zeros(c_size)
    return h0, c0
  
  def forward(self, Y):
    bsize = Y.size(0)
    Y = Y.permute(1, 0, 2)
    h0, c0 = self.get_initial(bsize)
    out, (h, c) = self.lstm(Y, (h0, c0))
    z = h.permute(1, 0, 2).mean(1)
    return z

class LSTMDecoder(nn.Module):
  """
  A general class for stacked lstm decoder
  """
  def __init__(self, output_length, output_size, hidden_size, latent_size, num_layers, directions):
    """
    inputs:
        output_length: length of the output (reconstructed) series
        output_size: dimension of the output (reconstructed) series
        latent_size: dimension of latent representation
        hidden_size: dimension of lstm hidden cells
        num_layers: number of stacked lstm
        directions: 1 for ordinary and 2 for bi-directional lstm
    """
    super().__init__()

    is_bidirectional = (directions == 2)
    
    self.lstm = nn.LSTM(latent_size, hidden_size=hidden_size,
                        bidirectional=is_bidirectional, num_layers=num_layers,
                        proj_size=output_size)

    self.latent_size = latent_size
    self.num_layers = num_layers
    self.directions = directions
    self.output_length = output_length
    self.output_size = output_size
    self.hidden_size = hidden_size
  
  def get_initial(self, batch_size):
    h_size = (self.directions * self.num_layers, batch_size, self.output_size)
    c_size = (self.directions * self.num_layers, batch_size, self.hidden_size)
    h0 = torch.zeros(h_size)
    c0 = torch.zeros(c_size)
    return h0, c0
  
  def forward(self, latent):
    bsize = latent.size(0)
  
    lstm_in = latent.unsqueeze(1).repeat(1, self.output_length, 1)
    lstm_in = lstm_in.permute(1, 0, 2)

    h0, c0 = self.get_initial(bsize)

    out, _ = self.lstm(lstm_in, (h0, c0))
    Y = 0.5 * (out[:, :, :self.output_size] + out[:, :, self.output_size:])
    Y = Y.permute(1, 0, 2)
    return Y
