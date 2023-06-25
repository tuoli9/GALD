#!/bin/bash -l

# GALD alone
python main.py --white vit --ra 0.7 --rd 0.1 
python main.py --white deit --ra 0.6 --rd 0.2 
python main.py --white tnt --ra 0.7 --rd 0.2 
python main.py --white pit --ra 0.5 --rd 0.1 

