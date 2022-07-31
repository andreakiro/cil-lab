"""
Helper methods for optimizer selection
To be used in the training algorithms
"""

import torch.optim as optim

def set_optimizer(model, optimizer, lr, wd, mom=0.9):
  optimizers = {
    'adam': optim.Adam(
      model.parameters(),
      lr=lr,
      weight_decay=wd,
    ),

    'adagrad': optim.Adagrad(
      model.parameters(),
      lr=lr,
      weight_decay=wd,
    ),

    'momentum': optim.SGD(
      model.parameters(),
      lr=lr,
      momentum=mom,
      weight_decay=wd,
    ),

    'rmsprop': optim.RMSprop(
      model.parameters(),
      lr=lr,
      momentum=mom,
      weight_decay=wd,
    ),
  }

  try:
    return optimizers[optimizer]
  except ValueError:
    return ValueError('Unknown optimizer.')