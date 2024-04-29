'''
Defined the setup to run the DeePMD training and MD

The training data was generated by NWChem and collected using ASE.
Now we need to train the DeePMD force field on the data. Subsequently
we can use the force field inside LAMMPS to run MD.

In order to accomplish the actions listed above we need:
    - Generate the input file for `dp` which is written in JSON
    - Run `dp train` to train the model
    - Compress the model
'''

import json
import os
import glob
import subprocess
import sys
from typing import List
from os import PathLike
from pathlib import Path
from deepdrivemd.config import BaseSettings

class DeePMDInput(BaseSettings):
    deepmd = {
        "model" : {
            "type_map" : ["h", "he", "li", "be", "b", "c", "n", "o", "f", "ne",
                          "na", "mg", "al", "si", "p", "s", "cl", "ar",
                          "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni",
                          "cu", "zn", "ga", "ge", "as", "se", "br", "kr",
                          "rb", "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd",
                          "ag", "cd", "in", "sn", "sb", "te", "i", "xe",
                          "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd",
                          "tb", "dy", "ho", "er", "tm", "yb", "lu", "hf", "ta", "w",
                          "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi", "po",
                          "at", "rn",
                          "fr", "ra", "ac", "th", "pa", "u", "np", "pu", "am", "cm",
                          "bk", "cf", "es", "fm", "md", "no", "lr", "rf", "db", "sg",
                          "bh", "hs", "mt", "ds", "rg", "cn", "nh", "fl", "mc", "lv",
                          "ts", "og"], # will be replaced by a compressed list
            "descriptor": {
                "type"          : "se_a", # was se_e3 but there were problems so try something from a working example
                "sel"           : "auto", # is this new?
                "rcut_smth"     : 3.0,
                "rcut"          : 6.0,
                # These hyperparameters came from a Silicon example:
                # https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-2/4-first-model
                #"neuron"        : [20,40,80],
                # These hyperparameters came from a Zn-protein example:
                # https://github.com/deepmodeling/deepmd-kit/blob/r2/examples/zinc_protein/zinc_se_a_mask.json
                "neuron"        : [32,32,64,128],
                #"type_one_side" : True, # does not exist for se_e3, se_at, or se_a_3be and will cause an error
                "axis_neuron"   : 16,
            },
            "fitting_net" : {
                # These hyperparameters came from a Silicon example:
                #"neuron"    : [80,80,80],
                # These hyperparameters came from a Zn-protein example:
                "neuron"    : [240,240,240],
                "resnet_dt" : True,
            },
        },
        "learning_rate" : {
            "start_lr"    : 0.002,
            "decay_steps" : 500,
        },
        "loss" : {
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "training" : {
            "stop_batch": 200000,
            "disp_file" : "lcurve.out",
            "disp_freq" : 2000,
            "save_freq" : 20000,
            "save_ckpt" : "model.ckpt",
            "validation_data" : {
                "systems"    : [],
                "batch_size" : "auto"
            },
            "training_data" : {
                "systems"    : [],
                "batch_size" : "auto"
            },
        }
    }

    def set_displ_file(self, newckpt_file: str) -> None:
        '''
        Set save_ckpt entry 

        When running multiple training instances in parallel
        you want to be sure to write the checkpoint files
        to separate files.
        '''
        self.deepmd["training"]["save_ckpt"] = str(newckpt_file)

    def set_checkpoint_file(self, newdisp_file: str) -> None:
        '''
        Set disp_file entry 

        When running multiple training instances in parallel
        you want to be sure to write the training progress 
        to separate files.
        '''
        self.deepmd["training"]["disp_file"] = str(newdisp_file)
        

    def set_type_map(self, newtypemap: List[str]) -> None:
        '''
        Store a new value for the "type_map" entry

        The original type_map includes all elements of the periodic table.
        This tends to cause problems with TensorFlow running out of memory.
        This function allows to store a compressed list instead, which
        hopefully circumvents memory problems.
        '''
        self.deepmd["model"]["type_map"] = newtypemap

    def set_sel(self, newsel: List[int]) -> None:
        '''
        Store a new value for the "sel" entry

        The "sel" entry specifies the maximum number of neighbors for every
        element type. Setting this too big impedes efficient training,
        setting it too small affects the accuracy of the model (the code
        drops the atoms that don't fit in the buffer?).

        To get a sensible estimate you need to collect the neighbor statistics
        on the training data:

          dp neighbor-stat -s <data> -r 8.0 -t <type_map>

        where:
        - "-s <data>" specifies the directory with training data
        - "-r 8.0" specifies the cutoff for atoms to be considered
        - "-t <type_map>" is the list of all relevant chemical symbols
        Sel is an upperbound. So if the typical cutoff is 6.0 running this
        command with a cutoff of 8.0 probably obtains a sensible result.
        '''
        self.deepmd["model"]["descriptor"]["sel"] = newsel

    def set_training_systems(self, newsystems: List[str]) -> None:
        '''
        Store a new list for the "training_data" "systems" entry

        The "systems" entry in "training_data" contains a list
        of paths to directories containing training data.
        '''
        self.deepmd["training"]["training_data"]["systems"] = newsystems

    def set_validation_systems(self, newsystems: List[str]) -> None:
        '''
        Store a new list for the "validation_data" "systems" entry

        The "systems" entry in "validation_data" contains a list
        of paths to directories containing validation data.
        '''
        self.deepmd["training"]["validation_data"]["systems"] = newsystems

    def dump_json(self, fpath: PathLike) -> None:
        '''
        Overload dump_json to store the contents of self.deepmd and not self.
        '''
        with open(fpath, mode="w") as fp:
            json.dump(self.deepmd, fp, indent=4, sort_keys=False)
        

def _list_max(l1: List[int], l2: List[int]) -> List[int]:
    '''
    Return the element wise maximum of two lists

    This is essentially an all-reduce with "max" on two lists.
    The implementation is very much inspired by
    https://stackoverflow.com/questions/35244791/finding-the-index-wise-maximum-values-of-two-lists
    '''
    return [max(*l) for l in zip(l1, l2)]

def _merge_type_maps(val_path: List[Path], trn_path: List[Path]) -> List[str]:
    '''
    Merge the type maps from various data directories

    We assume that the order of the chemical elements is unimportant.
    So we can simply read the various type_map.raw files, 
    add its elements to a dictionary (automatically ensures uniqueness),
    and the extract a list of its keys.
    '''
    all_path = val_path + trn_path
    symb_dict = {}
    for ipath in all_path:
        tm_path = Path(ipath,"type_map.raw")
        with open(tm_path,"r") as fp:
            type_map = fp.readline()
        type_map = type_map.split()
        for symb in type_map:
            symb_dict[symb] = 1
    return list(symb_dict.keys())

def gen_input(data_path: PathLike, json_path: PathLike) -> None:
    '''
    Generate DeePMD input file
    '''
    settings = DeePMDInput()
    val_path = Path(data_path,"**/validate_mol_*")
    trn_path = Path(data_path,"**/training_mol_*")
    validate_data = glob.glob(str(val_path),recursive=True)
    training_data = glob.glob(str(trn_path),recursive=True)
    settings.set_validation_systems(validate_data)
    settings.set_training_systems(training_data)
    settings.set_type_map(_merge_type_maps(validate_data,training_data))
    settings.dump_json(Path(json_path))

def train(train_path: PathLike, json_file: PathLike,
          model_file: PathLike = Path("model.pb"),
          compressed_model_file: PathLike = Path("compressed_model.pb"),
          ckpt_file: PathLike = None) -> None:
    '''
    Run the model training

    The basic command is

       dp train <json_file>
       dp freeze -o <model_file>
       dp compress -t <json_file> -i <model_file> -o <compressed_model_file>

    Note that effectively every separate training task has to run in a 
    separate directory. 

    - train_path is the directory where the training is supposed to run
    '''
    cwd = os.getcwd()
    trn_path = Path(train_path)
    if not trn_path.exists():
        os.makedirs(trn_path,exist_ok=True)
    elif not trn_path.is_dir():
        raise OSError(trn_path+" exists but is not a directory")
    os.chdir(trn_path)
    if ckpt_file:
        subprocess.run(["dp","train",str(json_file),"--init-model",str(ckpt_file)],stdout=sys.stdout)
    else:
        subprocess.run(["dp","train",str(json_file)],stdout=sys.stdout)
    subprocess.run(["dp","freeze","-o",str(model_file)],stdout=sys.stdout)
    subprocess.run(["dp","compress","-t",str(json_file),"-i",str(model_file),"-o",str(compressed_model_file)],stdout=sys.stdout)
    os.chdir(cwd)

