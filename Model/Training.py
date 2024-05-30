import yaml
import os 
import argparse

parser = argparse.ArgumentParser(description=" Training Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-C", "--Config_file",required=True,type=yaml, help="Config file")

with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
