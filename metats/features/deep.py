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


class GRUEncoder(nn.Module):
  """
  A general class for stacked GRU encoders
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

    self.gru = nn.GRU(input_size, hidden_size=hidden_size,
                      bidirectional=is_bidirectional, num_layers=num_layers)
    
    self.proj = nn.Linear(hidden_size, latent_size)

    self.latent_size = latent_size
    self.num_layers = num_layers
    self.directions = directions
    self.hidden_size = hidden_size
  
  def get_initial(self, batch_size):
    h_size = (self.directions * self.num_layers, batch_size, self.hidden_size)
    h0 = torch.zeros(h_size)
    return h0
  
  def forward(self, Y):
    bsize = Y.size(0)
    Y = Y.permute(1, 0, 2)
    h0 = self.get_initial(bsize)
    _, h = self.gru(Y, h0)
    z = h.permute(1, 0, 2).mean(1)
    z = self.proj(z)
    return z

class GRUDecoder(nn.Module):
  """
  A General class for stacked gru decoder
  """
  def __init__(self, output_length, output_size, hidden_size, latent_size, num_layers, directions):
    """
    inputs:
        output_length: length of the output (reconstructed) series
        output_size: dimension of the output (reconstructed) series
        latent_size: dimension of latent representation
        hidden_size: dimension of gru hidden cells
        num_layers: number of stacked gru
        directions: 1 for ordinary and 2 for bi-directional gru
    """
    super().__init__()

    is_bidirectional = (directions == 2)
    
    self.gru = nn.GRU(latent_size, hidden_size=hidden_size,
                      bidirectional=is_bidirectional, num_layers=num_layers)
    
    self.proj = nn.Linear(hidden_size, output_size)

    self.latent_size = latent_size
    self.num_layers = num_layers
    self.directions = directions
    self.output_length = output_length
    self.output_size = output_size
    self.hidden_size = hidden_size
  
  def get_initial(self, batch_size):
    h_size = (self.directions * self.num_layers, batch_size, self.hidden_size)
    h0 = torch.zeros(h_size)
    return h0
  
  def forward(self, latent):
    bsize = latent.size(0)
  
    gru_in = latent.unsqueeze(1).repeat(1, self.output_length, 1)
    gru_in = gru_in.permute(1, 0, 2)

    h0 = self.get_initial(bsize)

    out, _ = self.gru(gru_in, h0)
    Y = 0.5 * (out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:])
    Y = Y.permute(1, 0, 2)
    Y = self.proj(Y)
    return Y


class MLPEncoder(nn.Module):
  """
  A general class for MLP encoder
  """
  def __init__(self, input_size, input_length, latent_size, hidden_layers=(32,), activation=None):
    """
    inputs:
        input_size: dimension of input series
        input_length : length of input series
        latent_size: dimension of latent representation
        hidden_layers: a tuple of hidden layers dimension
        activation: a custom activation function which can be any PyTorch module supporting a backward pass,
                    if None passed, then the nn.Tanh() wiil be used
    """
    super().__init__()

    if activation == None:
      activation = nn.Tanh

    # input layer
    mlp_layers = [nn.Linear(input_size*input_length, hidden_layers[0])]
    mlp_layers.append(activation())
   
    # hidden layers
    for layer in range(len(hidden_layers)-1):
      mlp_layers.append(nn.Linear(hidden_layers[layer], hidden_layers[layer+1]))
      mlp_layers.append(activation())
    
    # output layer
    mlp_layers.append(nn.Linear(hidden_layers[-1], latent_size))
    mlp_layers.append(activation())

    self.mlp = nn.Sequential(*mlp_layers)
    
    self.latent_size = latent_size
  
  
  def forward(self, Y):
    Y = Y.flatten(1)
    z = self.mlp(Y)
    return z

class MLPDecoder(nn.Module):
  """
  A general class for MLP encoder
  """
  def __init__(self, input_size, input_length, latent_size, hidden_layers=(32), activation=None):
    """
    inputs:
        input_size: dimension of input series
        input_length : length of input series
        latent_size: dimension of latent representation
        hidden_layers: a tuple of hidden layers dimension
        activation: a custom activation function which can be any PyTorch module supporting a backward pass,
                    if None passed, then the nn.Tanh() wiil be used
    """
    super().__init__()

    if activation == None:
      activation = nn.Tanh
    
    # input layer
    mlp_layers = [nn.Linear(latent_size, hidden_layers[0])]
    mlp_layers.append(activation())
   
    # hidden layers
    for layer in range(len(hidden_layers)-1):
      mlp_layers.append(nn.Linear(hidden_layers[layer], hidden_layers[layer+1]))
      mlp_layers.append(activation())
    
    # output layer
    mlp_layers.append(nn.Linear(hidden_layers[-1], input_size*input_length))
    mlp_layers.append(activation())

    self.mlp = nn.Sequential(*mlp_layers)
    self.unflatten = nn.Unflatten(1, (input_length, input_size))

    self.latent_size = latent_size
  
  
  def forward(self, latent):
    Y = self.mlp(latent)
    Y = self.unflatten(Y)
    return Y

class AutoEncoder(nn.Module):
  """
  General AutoEncoder class
  """
  def __init__(self, encoder, decoder):
    """
    Args:
      encoder: a PyTorch module
      decoder: a PyTorch module
    """
    super().__init__()

    self.latent_size = encoder.latent_size

    self.encoder = encoder
    self.decoder = decoder

    self.rec_loss = nn.MSELoss()

  def encode(self, Y, inference=False):
    """
    encodes a mini batch of time series
    Args:
      Y : time series batch a PyTorch Tensor (batch_size x seires_length x series_dim)
      inference : if True, only forward pass will happen and the gradient won't be computed
    """
    if inference:
      self.encoder.eval()
      with torch.no_grad():
        return self.encoder(Y)
    else:
      return self.encoder(Y)

  def decode(self, latent):
    """
    decodes a mini batch of latent vectors
    Args:
      latent: a PyTorch Tensor (batch_size x latent_dim)
    """
    return self.decoder(latent)

  def loss(self, minibatch):
    latent = self.encode(minibatch['Y'], inference=False)
    reconstructed = self.decode(latent)
    loss = self.rec_loss(minibatch['Y'], reconstructed)
    return loss


class PyTorchTrainer():
  """
  Abstract trainer for PyTorch models
  """
  def __init__(self, model, batch_size=16, learning_rate=0.02):
    """
    Args:
      model : a PyTorch module
      batch_size: size of each mini batch
      learning_rate: optimizer's learning rate
    """
    self.model = model
    self.batch_size = 16
    self.learning_rate = learning_rate
    self.loss_callbacks = []

    self.initialize()
  
  def initial_optimizer(self):
    """
    initializing the optimizer
    """
    return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
  
  def initialize(self):
    self.optimizer = self.initial_optimizer()

  def register_loss_callback(self, fn):
    """
    register a callback to which the loss will be passed at each iteration
    """
    if callable(fn):
      self.loss_callbacks.append(fn)
    else:
      raise ValueError("The input is not a valid callable object")

  def apply_loss_callbacks(self, loss):
    """
    calling all registered callbacks
    """
    for fn in self.loss_callbacks:
      fn(loss)
  
  def get_mini_batch(self):
    """
    a method which must be implemented to provide a mini batch for model
    it's better to use a dictionary as mini batch
    """
    raise NotImplementedError("You must provide a method for sampling a batch")
  

  def step(self):
    """
    A single training step
    """
    # train mode
    self.model.train()
    # set thte accumulated gradient to zero 
    self.optimizer.zero_grad()
    # computing the loss for a mini batch
    mini_batch = self.get_mini_batch()
    loss = self.model.loss(mini_batch)
    # backprop and a single optimization step
    loss.backward()
    self.optimizer.step()
    # calling the registeres loss callbacks
    self.apply_loss_callbacks(loss.item())
