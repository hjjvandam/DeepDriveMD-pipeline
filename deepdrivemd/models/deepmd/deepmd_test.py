import deepmd
from pathlib import Path

data_path = Path("../../../sim/nwchem/test_dir")
cwd = os.getcwd()
json_file = Path(cwd,"input.json")
train1 = Path("./train-1")
train2 = Path("./train-2")

deepmd.gen_input(data_path,json_file)
deepmd.train(train1,json_file)
deepmd.train(train2,json_file)
deepmd.train(train1,json_file)
deepmd.train(train2,json_file)
