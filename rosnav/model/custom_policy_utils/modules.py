from torch import nn

def linear(params):
    return nn.Linear(params["in_features"], params["out_features"], params.get("bias", True))

def relu(params):
    return nn.ReLU(params.get("inplace", False))

modules = {
    "linear" : linear,
    "relu" : relu,
}