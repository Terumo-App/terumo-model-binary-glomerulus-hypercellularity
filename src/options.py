import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()


    def initialize(self):      
        self.parser.add_argument('--config_file', type=str, default='./config/default.yaml', help='# of test examples.')
    
    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt 