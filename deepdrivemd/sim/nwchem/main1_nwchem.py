'''
A test driver for the code in ase_nwchem.py

These tests make sure that
1. We can generate a valid NWChem input file with ASE
2. We can run an NWChem calculation for the energy and gradient
3. We can use ASE to extract the results of NWChem
4. We can store the results in a format suitable for DeePMD
'''

import os
import ase_nwchem
import glob
import sys
from pathlib import Path

# the NWCHEM_TOP environment variable needs to be set to specify
# where the NWChem executable lives.
nwchem_top = None
deepmd_source_dir = None
test_data = Path("../../../../../data/h2co/system")
test_pdb = Path(test_data,"h2co-unfolded.pdb")
test_inp = "h2co.nwi"
test_out = "h2co.nwo"
test_path = Path("./test_dir")
curr_path = Path("./")
test_path = Path(sys.argv[1])
if not test_path.exists():
    os.makedirs(test_path,exist_ok=True)
os.chdir(test_path)
print("Generate NWChem input files")
inputs_path = Path(test_path,"inputs.txt")
if not inputs_path.exists():
    # We haven't run any DFT calculations yet so generate input files
    # and store the list of inputs files
    # - grab a bunch of predefined input files
    # - perturb the initial molecular structure to generate more inputs
    inputs_cp = ase_nwchem.fetch_input(test_data)
    inputs_gn = ase_nwchem.perturb_mol(30,test_pdb)
    inputs = inputs_cp + inputs_gn
else:
    # We need to take new input files from the PDB structure generated
    # by the LAMMPS MD run
    pdbs_path = Path(sys.argv[2])
    inputs = []
    with open(pdbs_path, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        pdb_path = Path(line)
        input_path = Path(test_path,pdb_path.stem)
        input_name = Path(input_path,".nwi")
        ase_nwchem.nwchem_input(input_name,pdb_path)
        inputs.append(input_path)
with open("inputs.txt", "w") as f:
    for filename in inputs:
        print(str(filename), file=f)
print("Done NWChem input files")

# print("Run NWChem")
# for instance in inputs:
#     test_inp = instance.with_suffix(".nwi")
#     test_out = instance.with_suffix(".nwo")
#     ase_nwchem.run_nwchem(nwchem_top,test_inp,test_out)
# print("Extract NWChem results")
# test_dat = glob.glob("*.nwo")
# ase_nwchem.nwchem_to_raw(test_dat)
# print("Convert raw files to NumPy files")
# ase_nwchem.raw_to_deepmd(deepmd_source_dir)
# print("All done")
