import deepmd
import os
from pathlib import Path

cwd = os.getcwd()
data_path = Path(cwd,"../../sim/nwchem/test_dir")
json_file = Path(cwd,"input.json")
train1 = Path("./train-1")
train2 = Path("./train-2")
ckpt = Path("model.ckpt")

deepmd.gen_input(data_path,json_file)
deepmd.train(train1,json_file)
deepmd.train(train2,json_file)
deepmd.train(train1,json_file,ckpt_file=ckpt)
deepmd.train(train2,json_file,ckpt_file=ckpt)
