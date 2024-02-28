#!/opt/venv/bin/python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path config file", type=str)
args = parser.parse_args()

import rl4caribou

# training
#
from rl4caribou.utils import sb3_train    
model_save_id, train_options = sb3_train(args.file)

# hugging face
#
if 'repo' in train_options:
    from rl4caribou.utils import upload_to_hf
    try:
        upload_to_hf(args.file, "sb3/"+args.file, repo=train_options['repo'])
        upload_to_hf(model_save_id, "sb3/"+model_save_id+".zip", repo=train_options['repo'])
    except:
        print("Couldn't upload to hf!")