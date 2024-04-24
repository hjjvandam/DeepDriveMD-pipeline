"""Test the ase_lammps functionality."""
import ase_lammps
import glob
import os
from pathlib import Path

cwd = os.getcwd()
train = Path(cwd,"../../models/deepmd")
pdb = Path(cwd,"../../../data/h2co/system/h2co-unfolded.pdb")
data_dir = "pdbs"
test_dir = "test_dir"
freq = 100
os.mkdir(test_dir)
os.chdir(test_dir)

ase_lammps.lammps_input(pdb,train,freq)
ase_lammps.run_lammps()
failed, struct = ase_lammps.lammps_questionable(0.1,0.3,freq)
if failed:
    print("Reject trajectory")
else:
    print("Accept trajectory")
print(struct)
trajectory = Path(cwd,test_dir,"trj_lammps.dcd")
hdf5_basename = Path(cwd,test_dir,"trj_lammps")
ase_lammps.lammps_to_pdb(trajectory,pdb,struct,data_dir)
ase_lammps.lammps_contactmap(trajectory,pdb,hdf5_basename)

exit()

trajectories = glob.glob("scratch/trj_lammps*")
for trajectory in trajectories:
    print(trajectory)
    ase_lammps.lammps_to_pdb(trajectory,pdb,struct,data_dir)

