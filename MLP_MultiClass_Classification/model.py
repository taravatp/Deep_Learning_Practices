import torch
import torch.nn as nn
from torchsummary import summary


class MLP(nn.Module):
    def __init__(self, units: list, hidden_layer_activation='sigmoid', init_type=None):
        super(MLP, self).__init__()

        self.units = units
        self.n_layers = len(units)  # including input and output layers

        valid_activations = {'relu': nn.ReLU(),
                             'tanh': nn.Tanh(),
                             'sigmoid': nn.Sigmoid()
                             }

        if hidden_layer_activation is not None:
            activation = valid_activations[hidden_layer_activation]
        else:
            activation = None

        layers = []
        for layer in range(self.n_layers - 1):
            layers.append(nn.Linear(self.units[layer], self.units[layer + 1]))
            if layer != (self.n_layers - 2):
                if activation is not None:
                    # layers.append(nn.Dropout(0.2)) #uncomment this line to see the effect of droput layer
                    layers.append(activation)
                    

        self.mlp = nn.Sequential(*layers)
        self.init_type = init_type

    def forward(self, x):
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    batch, C, H, W = 128, 3, 64, 64
    zero_tensor = torch.zeros([batch, C, H, W], dtype=torch.float32)
    print(zero_tensor.shape)
    zero_tensor = torch.reshape(zero_tensor, (batch, -1))
    print(zero_tensor.shape)
    input_size = C * H * W
    num_classes = 6
    units = [input_size, 7, 7, num_classes]

    mlp = MLP(units=units, hidden_layer_activation='sigmoid')
    out = mlp(zero_tensor)
    assert out.shape == torch.Size(
        [batch, num_classes]), f'Model output size expected to be {torch.Size([batch, num_classes])}'
    summary(mlp, (batch, 3 * 64 * 64))
