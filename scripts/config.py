from torch import nn


def get_network_config(type='linear'):
    if type.lower() == 'linear':
        config = {
            '0': {
                'base_layer': nn.Linear,
                'kwargs': {
                    'in_features': 28 * 28,
                    'out_features': 24 * 24,
                }
            },
            '1': {
                'base_layer': nn.Linear,
                'kwargs': {
                    'in_features': 24 * 24,
                    'out_features': 300,
                }
            },
            '2': {
                'base_layer': nn.Linear,
                'kwargs': {
                    'in_features': 300,
                    'out_features': 100,
                }
            },
            '3': {
                'base_layer': nn.Linear,
                'kwargs': {
                    'in_features': 100,
                    'out_features': 20,
                }
            }
        }
    else:
        config = {
            '0': {
                'base_layer': nn.Conv2d,
                'kwargs': {
                    'in_channels': 1,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'bias': True
                }
            },
            '1': {
                'base_layer': nn.Conv2d,
                'kwargs': {
                    'in_channels': 16,
                    'out_channels': 32,
                    'kernel_size': 3,
                    'bias': True
                }
            },
            '2': {
                'base_layer': nn.Linear,
                'kwargs': {
                    'in_features': 32 * 24 * 24,
                    'out_features': 300,
                }
            },
            '3': {
                'base_layer': nn.Linear,
                'kwargs': {
                    'in_features': 300,
                    'out_features': 100,
                }
            },
            '4': {
                'base_layer': nn.Linear,
                'kwargs': {
                    'in_features': 100,
                    'out_features': 20,
                }
            }
        }

    return config