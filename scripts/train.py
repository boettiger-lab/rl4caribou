#!/opt/venv/bin/python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path config file", type=str)
parser.add_argument("-pb", "--progress_bar", help="Use  progress bar for training", type=bool, default=True)
parser.add_argument("-id", "--identifier", help="ID string for saving the agent", type=str, default="0")
args = parser.parse_args()

import rl4caribou

## normalizing work directory 
#
import os
abs_filepath = os.path.abspath(args.file)

# change directory to script's directory (since io uses relative paths)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# training
#
from rl4caribou.utils import sb3_train    
model_save_id, train_options = sb3_train(
    abs_filepath, 
    progress_bar=args.progress_bar, 
    identifier=args.identifier,
)
model_save_id = model_save_id + "_id_" + args.identifier

# hugging face
#
if 'repo' in train_options:
    from rl4caribou.utils import upload_to_hf
    try:
        upload_to_hf(abs_filepath, "sb3/"+args.file, repo=train_options['repo'])
        upload_to_hf(model_save_id, "sb3/"+model_save_id+".zip", repo=train_options['repo'])
    except:
        print("Couldn't upload to hf!")