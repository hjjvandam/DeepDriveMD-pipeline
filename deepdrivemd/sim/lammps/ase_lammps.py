"""Define the setup for a short LAMMPS MD simulation using DeePMD

Note that DeepDriveMD mainly thinks in terms of biomolecular structures.
Therefore it passes molecular structures around in PDB files. LAMMPS is
a general MD code that is more commonly used for materials (DL_POLY is
another example of such an MD code). Hence PDB files are strangers in 
LAMMPS's midst, but with the DeePMD force field this should be workable.

Approach:
    - Take the geometry in a PDB file
    - Take the force specification
    - Take the temperature
    - Take the number of time steps
    - Write the input file
    - Run the MD simulation
    - Check whether any geometries were flagged
    - Convert the trajectory for DeepDriveMD
"""

import os
import ase
from ase.calculators.lammpsrun import LAMMPS
from ase.io import iread
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.io.proteindatabank import read_proteindatabank, write_proteindatabank
from os import PathLike
from pathlib import Path
from typing import List

def lammps_run(pdb: PathLike, train: PathLike) -> None:
    """Create and run LAMMPS input file.

    Note that ASE gets the LAMMPS executable from the
    environment variable ASE_LAMMPSRUN_COMMAND.

    Arguments:
    pdb -- PDB file name for the molecular structure
    train -- path to the training directories
    """
    atoms = read_proteindatabank(pdb)
    temperature = 300.0
    outfreq = 100
    parameters = {
        "pair_style": f"deepmd {train}/train-1/compressed_model.pb {train}/train-2/compressed_model.pb {train}/train-3/compressed_model.pb  {train}/train-4/compressed_model.pb",
        "pair_coeff": ["* *"],
        "timestep": 0.001,
        "run": "10000 upto",
        "tmp_dir": "./scratch",
        #"fix": [f"fix_nvt all nvt temp {temperature} {temperature} $(100.0*dt)"],
        "thermo_args": ["step", "temp", "etotal", "ke", "pe", "atoms"],
        #"dump": f"myDump all atom {outfreq} trj_lammps.txt",
        #"dump": f"myDump all dcd {outfreq} trj_lammps.dcd", # LAMMPS can write DCD files but this is not accessible through ASE
        "binary_dump": False,
        "boundary": "p p p"
        }
    calc = LAMMPS(parameters=parameters,keep_tmp_files=True,tmpdir="./scratch",verbose=True)
    # ASE will complain about the line above and recommend using the line below BUT
    # calling set with the keyword "parameters" is a special case which directs
    # ASE to read those parameters from a file.
    #calc.set(parameters=parameters)
    #calc.set(tmpdir="./scratch")
    #atoms.set_calculator(LAMMPS(parameters=parameters,tmpdir="./scratch"))
    atoms.set_calculator(calc)
    try: 
        # This command is likely to fail but we need it to force ASE to actually run LAMMPS
        energy = atoms.get_potential_energy()
    except:
        # LAMMPS will likely have crashed we don't worry about that
        # instead we will sift through the data to see what useful 
        # information there is.
        pass

def lammps_questionable(force_crit: float) -> List[int]:
    """Return a list of all structures with large force mismatches."""
    structures = []
    with open("model_devi.out","r") as fp:
        # First line is just a line of headers
        line = fp.readline()
        # First line of real data
        line = fp.readline()
        while line:
            ln_list = line.split()
            struct_id = int(ln_list[0])
            error     = float(ln_list[4])
            if error > force_crit:
                structures.append(struct_id)
            line = fp.readline()
    return structures

class lammps_txt_trajectory:
    """A class to deal with LAMMPS trajectory data in txt format.

    A class instance manages a single trajectory file.
    - Creating an instance opens the trajectory file.
    - Destroying an instance closes the trajectory file.
    - Read will read the next timestep from the trajectory file.
    """
    def __init__(self, trj_file: PathLike, pdb_orig: PathLike):
        """Create a trajectory instance for trj_file.

        This constructor needs the PDB file from which the LAMMPS
        calculation was generated. In generating the LAMPS input
        the chemical element information was discarded. This means
        that the Atoms objects contain chemical nonsense information.
        By extracting the chemical element information from the 
        PDB file this information can be restored before returning
        the Atoms object.

        Arguments:
        trj_file -- the filename of the trajectory file
        pdb_orig -- the filename of the PDB file 
        """
        self.trj_file = trj_file
        self.trj_file_it = iread(trj_file,format="lammps-dump-text")
        atoms = read_proteindatabank(pdb_orig)
        self.trj_atomicno = atoms.get_atomic_numbers()
        self.trj_symbols = atoms.get_chemical_symbols()

    def next(self) -> ase.Atoms:
        atoms = next(self.trj_file_it,None)
        if atoms:
            atoms.set_atomic_numbers(self.trj_atomicno)
            atoms.set_chemical_symbols(self.trj_symbols)
        return atoms

def lammps_to_pdb(trj_file: PathLike, pdb_file: PathLike, indeces: List[int], data_dir: PathLike):
    """Write timesteps from the LAMMPS txt format trajectory to PDB files."""
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not Path(data_dir).is_dir():
        raise RuntimeError(f"{data_dir} exists but is not a directory")
    hashno = str(hash(trj_file))
    trajectory = lammps_txt_trajectory(trj_file,pdb_file)
    istep = -1
    for index in indeces:
        while index != istep:
            atoms = trajectory.next()
            istep += 1
            if not atoms:
                return
        # Found the desired time step so save it
        filename = Path(data_dir,f"atoms_{hashno}_{istep}.pdb")
        with open(filename,"w") as fp:
            write_proteindatabank(fp,atoms)

def lammps_to_dcd(trj_txt: PathLike, pdb_file: PathLike, trj_dcd: PathLike):
    """Convert a LAMMPS txt trajectory file into a DCD trajectory file."""
    universe = Universe(topology=pdb_file,trj_dcd)
    trajectory = lammps_txt_trajectory(trj_file,pdb_file)
    while True:
        atoms = trajectory.next()
        if not atoms:
            return
        positions = atoms.get_positions()
        symbol = atoms.get_chemical_symbols()

def lammps_to_ddmd(trj_file: PathLike, pdb_file: PathLike):
    """Convert the LAMMPS trajectory to DCD and HDF5.

    The LAMMPS trajectory file just contains the atom positions.
    It does not contain information about the chemical elements of
    the atoms. For that information we need the original PDB file.

    Arguments:
    trj_file -- the LAMMPS trajectory file
    pdb_file -- the original PDB file
    """
    pass
