from .farl_vit import FaRLVisualFeatures_TaskMOE

def backbone_entry(config):
    return globals()[config['type']](**config['kwargs'])