from configparser import ConfigParser
from os.path import join
import os
import json
import logging


def get_config_parser(filename='config_gan.cfg'):
    config = ConfigParser(allow_no_value=True)
    config.read(os.getcwd() + '/src/config/' + filename)
    logging.info('Getting config file: ' + os.getcwd() + '/src/config/' + filename)
    return config


def fill_config_with(config, config_parser, modifier, section, option):
    if config_parser.has_section(section) and config_parser.has_option(section, option):
        config[option.lower()] = modifier(config_parser.get(section, option))
    return config


def get_config(config_parser):
    config = {}
    
    config['dataroot'] = config_parser.get('dataset', 'DATAROOT')
    
    fill_config_with(config, config_parser, int, 'model', 'EMBEDDING_SIZE')

    fill_config_with(config, config_parser, json.loads, 'model', 'LAYERS')
  
    fill_config_with(config, config_parser, int, 'training', 'NUM_EPOCH')

    return config
