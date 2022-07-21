# Parses the parameters to a dictionary if module contains parameters
def parseParameters(module):
    if "params" in module:
        params = module["params"]
        parsedParameters = {}
        for param in params:
            parsedParameters[param["param"]] = param["value"]
        return parsedParameters
    else: 
        return {}
    

