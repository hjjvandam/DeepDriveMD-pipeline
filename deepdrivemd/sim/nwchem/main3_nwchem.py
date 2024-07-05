'''
A test driver for the code in ase_nwchem.py

These tests make sure that
1. We can generate a valid NWChem input file with ASE
2. We can run an NWChem calculation for the energy and gradient
3. We can use ASE to extract the results of NWChem
4. We can store the results in a format suitable for DeePMD
'''

import ase_nwchem
import glob
import os
from pathlib import Path
import random
import shutil
import sys

def prob_replace_pdb(total,success,good_pdb,start_pdb):
    """
    Probabilistically replace the MD start PDB file

    Normally the MD tasks start from the last structure that needed
    more training data from the previous MD run. This facilitates
    exploring more of the molecular phase space, but it can also
    take the calculation into rediculous regions of phase space.
    These regions are typically characterized with most if not all
    DFT calculations failing. This function probabilistically 
    replaces the starting PDB structure with a good structure
    based on the fraction of DFT calculations that fail.

    total - the total number of DFT calculations
    success - the number of successful DFT calculations
    good_pdb - a file with a known good structure
    start_pdb - the file with the MD starting structure
    """
    fails = total - success
    fraction_fails = (1.0*fails)/(1.0*total)
    rnd = random.uniform(0.0,1.0)
    if rnd <= fraction_fails:
        # Copy good_pdb over start_pdb
        path = shutil.copy(good_pdb,start_pdb)

# the NWCHEM_TOP environment variable needs to be set to specify
# where the NWChem executable lives.
nwchem_top = None
deepmd_source_dir = None
test_data = Path("../../../../data/h2co/system")
test_pdb = Path(test_data,"h2co-unfolded.pdb")
test_inp = "h2co.nwi"
test_out = "h2co.nwo"
test_path = Path("./test_dir")
curr_path = Path("./")
test_path = Path(sys.argv[1])
#os.mkdir(test_path)
os.chdir(test_path)
# print("Generate NWChem input files")
# inputs_cp = ase_nwchem.fetch_input(test_data)
# inputs_gn = ase_nwchem.perturb_mol(30,test_pdb)
# inputs = inputs_cp + inputs_gn
# print(inputs)
# print("Run NWChem")
# for instance in inputs:
#     test_inp = instance.with_suffix(".nwi")
#     test_out = instance.with_suffix(".nwo")
#     ase_nwchem.run_nwchem(nwchem_top,test_inp,test_out)
print("Extract NWChem results")
#test_dat = glob.glob("*.nwo")
# We just want to add the data from the last batch of DFT calculations.
test_dat = []
with open("inputs.txt", "r") as fp:
    lines = fp.readlines()
num_inputs = len(lines)
for line in lines:
    filename = line.strip()
    test_out = Path(filename).with_suffix(".nwo")
    if os.path.exists(test_out):
        test_dat.append(test_out)
num_success = len(test_dat)
prob_replace_pdb(num_inputs,num_success,test_pdb,Path("tmp.pdb"))
ase_nwchem.nwchem_to_raw(test_dat)
print("Convert raw files to NumPy files")
ase_nwchem.raw_to_deepmd(deepmd_source_dir)
print("All done")
