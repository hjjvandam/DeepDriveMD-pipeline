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
import glob
import itertools
import MDAnalysis as mda
import numpy as np
import operator
import os
import subprocess
from ase.calculators.lammpsrun import LAMMPS
from ase.data import atomic_masses, chemical_symbols
from ase.io import iread
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.io.proteindatabank import read_proteindatabank, write_proteindatabank
from MDAnalysis.analysis import distances, rms
from mdtools.analysis.order_parameters import fraction_of_contacts
from mdtools.writers import write_contact_map, write_point_cloud, write_fraction_of_contacts, write_rmsd
from mdtools.nwchem.reporter import OfflineReporter
from os import PathLike
from pathlib import Path
from typing import List

class Atoms:
    """Atoms class to trick MD-tools reporter."""
    def __init__(self,atom_lst: List):
        self._atoms = atom_lst
    def atoms(self):
        return self._atoms

class Simulation:
    """Simulation class to trick MD-tools reporter.

    MD-tools (https://github.com/hjjvandam/MD-tools/tree/nwchem) was
    originally designed to support OpenMM calculations in DeepDriveMD.
    I just want to run LAMMPS, and use MD-tools to convert the DCD file
    produced to a contact map file in HDF5 as required by DeepDriveMD. 
    At the same time I don't want to bring all of OpenMM into my
    environment just to be able to run this conversion. This Simulation
    class with the Atoms class above allow me to give the OfflineReporter
    what it needs without bringing all the OpenMM baggage along.
    """
    def __init__(self,pdb_file):
        self.pdb_file = Path(pdb_file)
        universe = mda.Universe(pdb_file,pdb_file)
        atomgroup = universe.select_atoms("all")
        atoms = [ag for ag in atomgroup]
        self.topology = Atoms(atoms)

def _sort_uniq(sequence):
    """Return a sorted sequence of unique instances.

    See https://stackoverflow.com/questions/2931672/what-is-the-cleanest-way-to-do-a-sort-plus-uniq-on-a-python-list
    """
    return map(operator.itemgetter(0),itertools.groupby(sorted(sequence)))

def lammps_input(pdb: PathLike, train: PathLike, freq: int) -> None:
    """Create the LAMMPS input file.

    The DeePMD models live in directories:

        {train}/train-*/compressed_model.pb

    The frequency specified here needs to match that in the
    trajectory checking. The trajectory will contain only 
    (#steps)/(freq) frames, whereas "model_devi.out" labels
    each structure by the original timestep.

    Arguments:
    pdb -- the PDB file with the structure
    train -- the path to the directory above the DeePMD models
    freq -- frequency of generating output
    """
    cwd = os.getcwd()
    temperature = 300.0
    steps = 10000
    freq = 100 # frequency of output in numbers of timesteps
    atoms = read_proteindatabank(pdb)
    pbc = atoms.get_pbc()
    if all(pbc):
        cell = atoms.get_cell()
    elif not any(pbc):
        cell = 2 * np.max(np.abs(atoms.get_positions())) * np.eye(3)
        atoms.set_cell(cell)
    lammps_data  = Path(cwd,"data_lammps_structure")
    lammps_input = Path(cwd,"in_lammps")
    lammps_trj   = Path(cwd,"trj_lammps.dcd")
    lammps_out   = Path(cwd,"out_lammps")
    # Taking compressed models out for now due to disk space limitations.
    # The compressed models are 10x larger than the uncompressed ones
    # (also raising questions about what compression means here).
    #deep_models = glob.glob(str(Path(train,"train-*/compressed_model.pb")))
    deep_models = glob.glob(str(Path(train,"train-*/model.pb")))
    with open(lammps_data,"w") as fp:
        write_lammps_data(fp,atoms)
    with open(lammps_input,"w") as fp:
        fp.write( "clear\n")
        fp.write( "atom_style   atomic\n")
        fp.write( "units        metal\n")
        fp.write( "atom_modify  sort 0 0.0\n\n")
        fp.write(f"read_data    {lammps_data}\n\n")
        pair_style = "pair_style   deepmd"
        for model in deep_models:
            pair_style += f" {model}"
        fp.write(f"{pair_style}\n")
        fp.write( "pair_coeff   * *\n\n")
        for i, cs in enumerate(_sort_uniq(atoms.get_chemical_symbols())):
            ii = i+1
            mass = atomic_masses[chemical_symbols.index(cs)]
            fp.write(f"mass {ii} {mass}\n")
        fp.write( "\n")
        fp.write( "timestep     0.001\n")
        fp.write(f"fix          fix_nvt  all nvt temp {temperature} {temperature} $(100.0*dt)\n")
        fp.write(f"dump         dump_all all dcd {freq} {lammps_trj}\n")
        fp.write( "thermo_style custom step temp etotal ke pe atoms\n")
        fp.write(f"thermo       {freq}\n")
        fp.write(f"run          {steps} upto\n")
        fp.write( "print        \"__end_of_ase_invoked_calculation__\"\n")
        fp.write(f"log          {lammps_out}\n")
            
def lammps_run() -> None:
    """Run a LAMMPS calculation.

    Note that ASE gets the LAMMPS executable from the
    environment variable ASE_LAMMPSRUN_COMMAND.
    """
    lammps_exe = Path(os.environ.get("ASE_LAMMPSRUN_COMMAND"))
    if not lammps_exe:
        raise RuntimeError("lammps_run: ASE_LAMMPSRUN_COMMAND undefined")
    if not Path(lammps_exe).is_file():
        raise RuntimeError("lammps_run: ASE_LAMMPSRUN_COMMAND("+lammps_exe+") is not a file")
    with open("in_lammps","r") as fp_in:
        subprocess.run([lammps_exe],stdin=fp_in)

def lammps_questionable(force_crit_lo: float, force_crit_hi: float, freq: int) -> List[int]:
    """Return a list of all structures with large force mismatches.

    There are two criteria. If the difference in the forces exceeds
    the lower criterion then the corresponding structure should be
    added to the training set. If the difference exceeds the higher
    criterion for any point then the errors are so severe that the
    trajectory should be considered non-physical. So its structures
    should not be used in the DeepDriveMD loop.

    Arguments:
    force_crit_lo -- the lower force criterion
    force_crit_hi -- the higher force criterion
    """
    structures = []
    failed = False
    with open("model_devi.out","r") as fp:
        # First line is just a line of headers
        line = fp.readline()
        # First line of real data
        line = fp.readline()
        while line:
            ln_list = line.split()
            struct_id = int(ln_list[0])
            error     = float(ln_list[4])
            if error > force_crit_lo:
                if struct_id % freq != 0:
                    raise RuntimeError("lammps_questionable: frequency mismatch")
                structures.append(int(struct_id/freq))
            if error > force_crit_hi:
                failed = True
            line = fp.readline()
    return (failed, structures)

#class lammps_txt_trajectory:
#    """A class to deal with LAMMPS trajectory data in txt format.
#
#    A class instance manages a single trajectory file.
#    - Creating an instance opens the trajectory file.
#    - Destroying an instance closes the trajectory file.
#    - Read will read the next timestep from the trajectory file.
#    """
#    def __init__(self, trj_file: PathLike, pdb_orig: PathLike):
#        """Create a trajectory instance for trj_file.
#
#        This constructor needs the PDB file from which the LAMMPS
#        calculation was generated. In generating the LAMPS input
#        the chemical element information was discarded. This means
#        that the Atoms objects contain chemical nonsense information.
#        By extracting the chemical element information from the 
#        PDB file this information can be restored before returning
#        the Atoms object.
#
#        Arguments:
#        trj_file -- the filename of the trajectory file
#        pdb_orig -- the filename of the PDB file 
#        """
#        self.trj_file = trj_file
#        self.trj_file_it = iread(trj_file,format="lammps-dump-text")
#        atoms = read_proteindatabank(pdb_orig)
#        self.trj_atomicno = atoms.get_atomic_numbers()
#        self.trj_symbols = atoms.get_chemical_symbols()
#
#    def next(self) -> ase.Atoms:
#        atoms = next(self.trj_file_it,None)
#        if atoms:
#            atoms.set_atomic_numbers(self.trj_atomicno)
#            atoms.set_chemical_symbols(self.trj_symbols)
#        return atoms

def lammps_to_pdb(trj_file: PathLike, pdb_file: PathLike, indeces: List[int], data_dir: PathLike):
    """Write timesteps from the LAMMPS DCD format trajectory to PDB files."""
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not Path(data_dir).is_dir():
        raise RuntimeError(f"{data_dir} exists but is not a directory")
    hashno = str(abs(hash(trj_file)))
    universe = mda.Universe(pdb_file,trj_file)
    selection = universe.select_atoms("all")
    ii = 0
    istep_trj = -1
    istep_lst = indeces[ii] 
    for ts in universe.trajectory:
        istep_trj +=1
        while istep_lst <  istep_trj:
            ii += 1
            if ii >= len(indeces):
                # We are done
                return
            istep_lst = indeces[ii]
        print(f"lst, trj: {istep_lst} {istep_trj}")
        if istep_lst == istep_trj:
            # Convert this structure to PDB
            filename = Path(data_dir,f"atoms_{hashno}_{istep_trj}.pdb")
            with mda.Writer(filename,universe.trajectory.n_atoms) as wrt:
                wrt.write(selection)

def lammps_contactmap(trj_file: PathLike, pdb_file: PathLike, hdf5_file: PathLike):
    """Write timesteps from the LAMMPS DCD format trajectory to PDB files."""
    hashno = str(abs(hash(trj_file)))
    trj = mda.Universe(pdb_file,trj_file)
    pdb = mda.Universe(pdb_file,pdb_file)
    sim = Simulation(pdb_file)
    selection = trj.select_atoms("all")
    atoms = [ag.name for ag in selection]
    report_steps = 100
    frames_per_h5 = int(10000/report_steps)

    reporter = OfflineReporter(
                   hdf5_file,report_steps,frames_per_h5=frames_per_h5,
                   wrap_pdb_file=None,reference_pdb_file=pdb_file,
                   openmm_selection=atoms,mda_selection="all",
                   threshold=8.0,
                   contact_map=False,point_cloud=True,fraction_of_contacts=False)
    for ts in trj.trajectory:
        reporter.report(sim,ts)
