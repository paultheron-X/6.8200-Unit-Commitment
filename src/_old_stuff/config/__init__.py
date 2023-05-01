from .utils import get_config, get_config_parser


"""
Method: 
    > When you want to add a new variable to the config file
    
    1. Add it to the cfg file
    2. update the get_config in utils
    3. update yr program
    
"""

def return_config(config_name):
    if config_name == 'tpl':
        return get_config(get_config_parser('config_tpl.cfg'))
    else:
        raise ValueError('Invalid config name')
