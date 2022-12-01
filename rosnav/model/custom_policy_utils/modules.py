from torch import nn

def linear(params):
    return nn.Linear(params["in_features"], params["out_features"], params.get("bias", True))

def relu(params):
    return nn.ReLU(params.get("inplace", False))

def conv1d(params):
    return nn.Conv1d(params["in_channels"], params["out_channels"], params["kernel_size"], params.get("stride", 1), params.get("padding", 0), params.get("dilation", 1), params.get("groups", 1), params.get("bias", True), params.get("padding_mode", 'zeros'))

def conv2d(params):
    return nn.Conv2d(params["in_channels"], params["out_channels"], params["kernel_size"], params.get("stride", 1), params.get("padding", 0), params.get("dilation", 1), params.get("groups", 1), params.get("bias", True), params.get("padding_mode", 'zeros'))

def conv3d(params):
    return nn.Conv3d(params["in_channels"], params["out_channels"], params["kernel_size"], params.get("stride", 1), params.get("padding", 0), params.get("dilation", 1), params.get("groups", 1), params.get("bias", True), params.get("padding_mode", 'zeros'))

def tanh(params):
    return nn.Tanh()

modules = {
    "relu" : relu,
    "tanh" : tanh,
    "linear" : linear,
    "conv1d" : conv1d,
    "conv2d" : conv2d,
    "conv3d" : conv3d,
}
