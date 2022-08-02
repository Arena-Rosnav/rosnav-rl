from torch import nn

def linear(params):
    return nn.Linear(params["in_features"], params["out_features"], params.get("bias", True))

def relu(params):
    return nn.ReLU(params.get("inplace", False))

def conv1d(params):
    return nn.Conv1d(params["in_channels"], params["out_channels"], params["kernel_size"], params.get["stride"], params.get["padding"], params.get["padding_mode"], params.get["dilation"], params.get["groups"], params.get["bias"])

def conv2d(params):
    return nn.Conv1d(params["in_channels"], params["out_channels"], params["kernel_size"], params.get["stride"], params.get["padding"], params.get["padding_mode"], params.get["dilation"], params.get["groups"], params.get["bias"])

def conv3d(params):
    return nn.Conv1d(params["in_channels"], params["out_channels"], params["kernel_size"], params.get["stride"], params.get["padding"], params.get["padding_mode"], params.get["dilation"], params.get["groups"], params.get["bias"])

def tanh(params):
    return nn.Tanh

modules = {
    "relu" : relu,
    "tanh" : tanh,
    "linear" : linear,
    "conv1d" : conv1d,
    "conv2d" : conv2d,
    "conv3d" : conv3d,
}