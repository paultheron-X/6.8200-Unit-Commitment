import os
import re
import torch.nn as nn
import sys
import logging

class BaseModel(nn.Module):
    
    @property
    def name(self):
        return 'BaseModel'

    def forward(self, x):
        return NotImplementedError('forward method is not implemented')
    
    def save(self, path: str) -> None:
        return NotImplementedError('save method is not implemented')
    
    def load(self, path: str) -> None:
        return NotImplementedError('load method is not implemented')