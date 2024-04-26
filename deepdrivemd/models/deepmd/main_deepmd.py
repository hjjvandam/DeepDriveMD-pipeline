import deepmd
import os
import sys
from pathlib import Path

cwd = os.getcwd()
data_path = Path(cwd,"../../sim/nwchem/test_dir")
data_path = sys.argv[1]
train = sys.argv[2]
json_file = Path(train,"input.json")
ckpt = Path("model.ckpt")

deepmd.gen_input(data_path,json_file)
if not train.exists():
    deepmd.train(train,json_file)
else:
    deepmd.train(train,json_file,ckpt_file=ckpt)

