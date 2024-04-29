"""Test the ase_lammps functionality."""
import ase_lammps
import glob
import os
from pathlib import Path
import sys

cwd = os.getcwd()
test_dir = Path(sys.argv[1])
pdb = Path(sys.argv[2],"*.pdb")
train = Path(sys.argv[3])
pdbs = glob.glob(str(pdb))
pdb = Path(pdbs[0])
data_dir = "pdbs"
print("Begin LAMMPS run")
freq = 100
if not test_dir.exists():
    os.makedirs(test_dir,exist_ok=True)
os.chdir(test_dir)

ase_lammps.lammps_input(pdb,train,freq)
ase_lammps.run_lammps()
failed, struct = ase_lammps.lammps_questionable(0.1,0.3,freq)
success = not failed
with open("lammps_success.txt", "w") as fp:
    print(success, file=fp)
if failed:
    print("Reject trajectory")
else:
    print("Accept trajectory")
print(struct)
trajectory = Path(cwd,test_dir,"trj_lammps.dcd")
hdf5_basename = Path(cwd,test_dir,"trj_lammps")
ase_lammps.lammps_to_pdb(trajectory,pdb,struct,data_dir)
ase_lammps.lammps_contactmap(trajectory,pdb,hdf5_basename)
print("Done  LAMMPS run")
