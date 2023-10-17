import random

import numpy as np
import torch


class Map(dict):
    """
    Class that extends dictionry to allow for dot access of keys.

    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    m.first_name # Eduardo
    m['first_name'] # Eduardo
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def format_figure(fig, **kwargs):
    fig.update_layout(plot_bgcolor='white', **kwargs)
    fig.update_xaxes(showline=True,linewidth=1.5,linecolor='black',tickfont=dict(size=20),
                        mirror=True,ticks='outside',showgrid=False,titlefont=dict(size=24)
                        )
    fig.update_yaxes(showline=True,linewidth=1.5,linecolor='black',
                        mirror=True,ticks='outside',showgrid=False,tickfont=dict(size=20),titlefont=dict(size=24))
    return fig


def change_figure_colors(fig, colors):
    for trace_idx, color in enumerate(colors):
        fig.data[trace_idx].marker.color = color
        if fig.data[trace_idx].type != 'bar':
            fig.data[trace_idx].line.color = color
    return fig

def safe_softmax(t, eps=1e-6, **kwargs):
    """
    Softmax function that always sums to 1 or less. Handles occasional numerical errors in torch's softmax.
    """
    return torch.softmax(t, **kwargs)-eps


def set_random_seed(params):
    try:
        seed = params.seed
    except:
        seed = params
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def thermometer_encode(labels, num_classes):
    feats = torch.zeros([labels.shape[0], num_classes])
    for i, label in enumerate(labels):
        feats[i, :int(label)] = 1
    return feats


def one_hot_encode(labels, num_classes):
    """
    One hot encode labels and convert to tensor.
    """
    return torch.tensor((np.arange(num_classes) == labels[..., None]).astype(float),dtype=torch.float32)