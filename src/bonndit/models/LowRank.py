import torch as T

class LowRankModel(T.nn.Module):
    def __init__(self, layers, dropout):
        super(LowRankModel, self).__init__()
        self.layers = T.nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(T.nn.Linear(layers[i], layers[i+1]))
            self.layers.append(T.nn.GELU())
     #       self.layers.append(T.nn.BatchNorm1d(layers[i+1]))
            self.layers.append(T.nn.Dropout(dropout))
        self.layers = self.layers[:-2]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
 