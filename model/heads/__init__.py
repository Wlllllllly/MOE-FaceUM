from .task_specific_heads import TaskSpecificHeadsHolder

def heads_holder_entry(config):
    return globals()[config['type']](**config['kwargs'])