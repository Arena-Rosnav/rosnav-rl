import json
from torch import nn
from .modules import modules

# Parses the parameters to a dictionary if module contains parameters
def parseParameters(module):
    if not "params" in module:
        return {}
       
    params = module["params"]
    parsedParameters = {}
    for param in params:
        parsedParameters[param["param"]] = param["value"]
    return parsedParameters
    
def readJson(path):
    with open(path, 'r') as f:
        return json.load(f)

def createBodyNetwork(data):
    body_net=nn.Sequential()

    # Number of the module to be added to the NN
    moduleNumber = 0

    # Iterate over each module given in the json in the policy
    for module in data["policy"]:

        # Parse parameters for module
        parsedParameters = parseParameters(module)

        # Add each module to the body network with corresponding number of module
        body_net.add_module(f'{moduleNumber}', modules[module["name"].lower()](parsedParameters))

        moduleNumber += 1

    return body_net
