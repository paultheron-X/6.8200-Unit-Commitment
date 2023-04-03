from .base_model import BaseModel

import logging

class CustomModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.net = None
        
    def forward(self, x):
        return x
    
    def save(self, path: str) -> None:
        pass
    
    def load(self, path: str) -> None:
        pass