import torch


def resolve_device(device_name: str) -> torch.device:
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is not available on this machine.')

    return torch.device(device_name)


def get_torch_load_kwargs(device: torch.device) -> dict:
    if device.type == 'cuda':
        return {}

    return {'map_location': device}
