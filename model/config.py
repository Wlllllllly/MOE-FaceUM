import yaml
import numpy as np
from easydict import EasyDict as edict
import re
from core.utils import printlog

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


class Config(object):

    def __init__(self, args, rank, local_rank, world_size):

        config_path = args.config
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=loader)

        self.common = edict(self.config["common"])
        
        





        




        