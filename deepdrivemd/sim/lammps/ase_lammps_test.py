"""Test the ase_lammps functionality."""
import ase_lammps
import os
from pathlib import Path

cwd = os.getcwd()
train = Path(cwd,"../../models/deepmd")
pdb = Path(cwd,"../../../data/h2co/system/h2co-unfolded.pdb")

ase_lammps.lammps_run(pdb,train)

