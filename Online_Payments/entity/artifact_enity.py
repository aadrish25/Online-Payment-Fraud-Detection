import os,sys
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_path:str
    test_path:str