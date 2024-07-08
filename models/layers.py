import torch


def get_norm_1d(norm_type, fin):
    norm_type = norm_type.lower()
    if norm_type is None or norm_type == 'none':
        return None
    elif norm_type == 'in':
        return torch.nn.InstanceNorm1d(fin)
    elif norm_type == 'batch':
        return torch.nn.BatchNorm1d(fin)
    elif norm_type == 'layer':
        return torch.nn.LayerNorm(fin)
    else:
        raise ValueError
