import json
from torch import nn
from .modules import modules

# Parses the parameters to a dictionary if module contains parameters
def parse_parameters(module):
    if not "params" in module:
        return {}
       
    params = module["params"]
    parsed_parameters = {}
    for param in params:
        parsed_parameters[param["param"]] = param["value"]
    return parsed_parameters
    

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)
        

def create_body_network(data):
    body_net = nn.Sequential()

    # Number of the module to be added to the NN
    module_number = 0

    # Iterate over each module given in the json in the policy
    for module in data["policy"]:

        # Parse parameters for module
        parsed_parameters = parse_parameters(module)

        # Add each module to the body network with corresponding number of module
        body_net.add_module(f'{module_number}', modules[module["name"].lower()](parsed_parameters))

        module_number += 1

    return body_net
