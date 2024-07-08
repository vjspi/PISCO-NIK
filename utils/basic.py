import re

import numpy
import yaml
import os
import importlib
import fnmatch
import torch
import numpy as np


def float32_serializer(obj):
    if isinstance(obj, np.float32):
        return round(float(obj), 3)  # Convert float32 to standard float for JSON serialization
    raise TypeError(f"Type {type(obj)} not serializable")

def numpy2torch(array, device="cpu"):
    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array).to(device)
    elif torch.is_tensor(array):
        return array.to(device)
    elif array is None:
        return None
    else:
        AssertionError()

def torch2numpy(tensor):
    if isinstance(tensor, numpy.ndarray):
        return tensor
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def find_subfolder(base_folder, target_string):
    for root, dirs, files in os.walk(base_folder):
        for dir_name in fnmatch.filter(dirs, target_string):
            return os.path.join(root, dir_name)
    return None

def import_module(module_name: str, class_name: str):
    """
    Import a module (class_name) from a module_name (path).

    :param module_name:
        Name of the module from where a class should be imported.
    :param class_name:
        Name of the class to import.
    :return:
        class_: type
            Reference to the imported class.
    """
    module_ = importlib.import_module(module_name)
    class_ = getattr(module_, class_name)
    return class_

## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

def parse_config(path=None, data=None, tag='!ENV'):
    """
    Load a yaml configuration file and resolve any environment variables
    The environment variables must have !ENV before them and be in this format
    to be parsed: ${VAR_NAME}.
    E.g.:
    database:
        host: !ENV ${HOST}
        port: !ENV ${PORT}
    app:
        log_path: !ENV '/var/${LOG_PATH}'
        something_else: !ENV '${AWESOME_ENV_VAR}/var/${A_SECOND_AWESOME_VAR}'
    :param str path: the path to the yaml file
    :param str data: the yaml data itself as a stream
    :param str tag: the tag to look for
    :return: the dict configuration
    :rtype: dict[str, T]
    """
    # pattern for global vars: look for ${word}
    pattern = re.compile('.*?\${(\w+)}.*?')
    loader = yaml.SafeLoader

    ## register the tag handler
    loader.add_constructor('!join', join)

    # the tag will be used to mark where to start searching for the pattern
    # e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
    loader.add_implicit_resolver(tag, pattern, None)

    def constructor_env_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', os.environ.get(g, g)
                )
            return full_value
        return value

    loader.add_constructor(tag, constructor_env_variables)

    if path:
        with open(path) as conf_data:
            return yaml.load(conf_data, Loader=loader)
    elif data:
        return yaml.load(data, Loader=loader)
    else:
        raise ValueError('Either a path or data should be defined as input')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

