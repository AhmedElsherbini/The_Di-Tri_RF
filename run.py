# Import the argparse library
import argparse
import os
from pca import PCA_model
from train import *
from test import *

#####################
# Create the parser
my_parser = argparse.ArgumentParser(description='Welcome to our tool, make sure your files exsit in the data folder!')
print("$ python run.py -m train -i ./data/x_data.csv")
# Add the arguments



my_parser.add_argument('-m','--mode',
                       action='store',
                        metavar='mode',
                       type=str,
                       help='PCA or train or test')

my_parser.add_argument('-i','--input',
                       action='store',
                       metavar='input',
                       type=str,
                       help='the path to your file')

# Execute the parse_args() method
args = my_parser.parse_args()
###############
fg  = args.mode
f_name = args.input
################
if (fg == "PCA"):
      PCA_model(f_name)

elif (fg == "train"):
      trainML_model(f_name)
      
elif(fg == "test"):
    testML_model(f_name)  


