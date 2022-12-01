import json
from torch import nn
from .modules import modules

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)
        

def create_body_network(data):
    body_net = nn.Sequential()

    # Number of the module to be added to the NN
    module_number = 0

    # Iterate over each module given in the json in the policy
    for module in data["policy"]:

        # Add each module to the body network with corresponding number of module
        body_net.add_module(f'{module_number}', modules[module["type"].lower()](module))

        module_number += 1

    return body_net
