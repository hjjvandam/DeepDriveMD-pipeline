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

import ase
from ase.calculators.lammpsrun import LAMMPS
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.io.proteindatabank import read_proteindatabank, write_proteindatabank

def lammps_run(pdb: PathLike, train: PathLike) -> None:
    """Create LAMMPS input file.

    Arguments:
    pdb -- PDB file name for the molecular structure
    train -- path to the training directories
    """
    atoms = read_proteindatabank(pdb)
    #data_file = Path(inpf,".lammps-data")
    #write_lammps_data(data_file,atoms)
    parameters = {
        "pair_style": f"deepmd {train}/train-1/compressed_model.pb {train}/train-2/compressed_model.pb {train}/train-3/compressed_model.pb  {train}/train-4/compressed_model.pb",
        "pair_coeff": "* *",
        "temperature": 300.0,
        "pressure": 1.0,
        "timestep": 0.001,
        "run": "10000 upto"
        }
    atoms.set_calculator(LAMMPS(parameters=parameters,tmpdir="./scratch"))
    energy = atoms.get_potential_energy()
