import numpy as np
import os
from pathlib import Path
import sfparamgen as sfp
import subprocess
import sys
import typing

def read_elements(fname: Path) -> list[str]:
    '''Read the elements in the training set

    The chemical elements are list in the commend line in the input.data
    file. The elements are the string in round brackets. This string
    is extracted, converted into a list and returned.
    '''
    with open(fname,'r') as fp:
        while True:
            entry = fp.readline()
            if entry.startswith("comment"):
                tmp1 = entry.split("(")[1]
                tmp2 = tmp1.split(")")[0]
                elements = tmp2.split()
                return elements

def gen_symfunc(elements: list[str], fname: Path, r_cutoff: float = 6.0) -> None:
    '''Generate the symmetry functions

    The N2P2 approach needs symmetry functions as inputs to the NNP
    setup. The functions are generated based on:
    - the chemical elements involved
    - the design rules preferred
    The symmetry functions are written out to one of the N2P2
    input files. The input file name is given in fname and this function
    appends the symmetry functions to that file.
    There are two sets of rules derived from two papers:
    - 'gastegger2018' from https://doi.org/10.1063/1.5019667
    - 'imbalzano2018' from https://doi.org/10.1063/1.5024611
    '''
    gen = sfp.SymFuncParamGenerator(elements,r_cutoff)
    rule = 'gastegger2018'
    rule = 'imbalzano2018'
    mode = 'center'
    mode = 'shift'
    r_lower = 0.01
    r_upper = r_cutoff
    if rule == 'imbalzano2018':
        r_lower = None
        r_upper = None
    nb_param_pairs=5
    gen.generate_radial_params(rule=rule,mode=mode,nb_param_pairs=nb_param_pairs)

    with open(fname,'a') as fp:
        gen.symfunc_type = 'radial'
        gen.write_settings_overview(fileobj=fp)
        gen.write_parameter_strings(fileobj=fp)
        gen.symfunc_type = 'weighted_radial'
        gen.write_settings_overview(fileobj=fp)
        gen.write_parameter_strings(fileobj=fp)
        gen.zetas = [1.0,6.0]
        gen.symfunc_type = 'angular_narrow'
        gen.write_settings_overview(fileobj=fp)
        gen.write_parameter_strings(fileobj=fp)
        gen.symfunc_type = 'angular_wide'
        gen.write_settings_overview(fileobj=fp)
        gen.write_parameter_strings(fileobj=fp)
        gen.symfunc_type = 'weighted_angular'
        gen.write_settings_overview(fileobj=fp)
        gen.write_parameter_strings(fileobj=fp)

def run_scaling():
     '''Run nnp-scaling

     One needs to run nnp-scaling to compute symmetry function
     statistics that the training will use. 
     '''
     n2p2_root = os.environ.get("N2P2_ROOT")
     if not n2p2_root:
         scaling_exe = "nnp-scaling"
     else:
         scaling_exe = Path(n2p2_root) / "bin" / "nnp-scaling"
     scaling_exe = str(scaling_exe)
     nnp_nproc = 1
     with open("nnp-scaling.out","w") as fpout:
         subprocess.run([scaling_exe,"100"],stdout=fpout,stderr=subprocess.STDOUT)

def run_training():
     '''Run nnp-training
     '''
     n2p2_root = os.environ.get("N2P2_ROOT")
     if not n2p2_root:
         training_exe = "nnp-train"
     else:
         training_exe = Path(n2p2_root) / "bin" / "nnp-train"
     training_exe = str(training_exe)
     nnp_nproc = 1
     with open("nnp-training.out","w") as fpout:
         subprocess.run([training_exe],stdout=fpout,stderr=subprocess.STDOUT)

def write_input(elements: list[str], cutoff_type: int, cutoff_alpha: float) -> None:
     '''Write an input file

     This function writes an input file for the N2P2 tools. 

     Some of the parameters are case specific so the corresponding
     values need to be passed by the function arguments. This is 
     particularly true for the chemical elements in the system of
     interest. 

     Other characteristics we may want to set are the cutoff type
     and the cutoff radius. More on this below.

     Finally, N2P2 uses random number generators but the seed is
     specified in the input file. Here we want to use the NNP in a mode
     that is similar to DeePMD. I.e. we want to train models with the
     same hyperparameters but different initial weights to get a sense
     of the parameter uncertainty after training. That means that for
     every model we train we need a unique seed. Python's random number
     generator can be initialized with a hardware entropy pool. We'll
     use this approach to pick random random number generator seeds.

     N2P2 supports different cutoff types which are enumerated as:
     - CT_HARD  (0): No cutoff(?)
     - CT_COS   (1): (cos(pi*x)+1)/2
     - CT_TANHU (2): tanh^3(1 - r/r_c)
     - CT_TANH  (3): tanh^3(1 - r/r_c), except if r=0 then 1
     - CT_EXP   (4): exp(1 - 1/(1-x*x))
     - CT_POLY1 (5): (2x - 3)x^2 + 1
     - CT_POLY2 (6): ((15 - 6x)x - 10) x^3 + 1
     - CT_POLY3 (7): (x(x(20x - 70) + 84) - 35)x^4 + 1
     - CT_POLY4 (8): (x(x((315 - 70x)x - 540) + 420) - 126)x^5 + 1
     See: n2p2/src/libnnp/CutoffFunction.h

     In general we follow the suggestions in
     https://github.com/CompPhysVienna/n2p2/blob/master/examples/input.nn.recommended
     '''
     num_elm = len(elements)
     if num_elm < 1:
         raise RuntimeError(f"N2P2 write_input: Invalid number of chemical elements: {num_elm}")
     with open("input.nn","w") as fp:
         fp.write(f"number_of_elements {num_elm}\n")
         fp.write( "elements")
         for element in elements:
             fp.write(f" {element}")
         fp.write( "\n")
         fp.write(f"cutoff_type {str(cutoff_type)} {str(cutoff_alpha)}\n")
         fp.write( "scale_symmetry_functions_sigma\n")
         fp.write( "scale_min_short 0.0\n")
         fp.write( "scale_max_short 1.0\n")
         fp.write( "global_hidden_layers_short 2\n")
         fp.write( "global_nodes_short 15 15\n")
         fp.write( "global_activation_short p p l\n")
         fp.write( "use_short_forces\n")
         # The random_seed is mentioned here so we don't forget it.
         # All the parameters printed out here are general for this case.
         # The random_seed need to be set separately for every training input
         # and needs to be unique among all training runs.
         # So after generating the generic input files we append a different
         # random_seed for every training input file. Which is probably the
         # easiest way of handling this situation.
         fp.write( "#random_seed - we'll append that at the end\n")
         fp.write( "epochs 10\n")
         fp.write( "normalize_data_set force\n")
         fp.write( "updater_type 1\n")
         fp.write( "parallel_mode 0\n")
         fp.write( "jacobian_mode 1\n")
         fp.write( "update_strategy 0\n")
         fp.write( "selection_mode 2\n")
         fp.write( "task_batch_size_energy 1\n")
         fp.write( "task_batch_size_force 1\n")
         fp.write( "memorize_symfunc_results\n")
         fp.write( "test_fraction 0.1\n")
         fp.write( "force_weight 1.0\n")
         fp.write( "short_energy_fraction 1.000\n")
         fp.write( "force_energy_ratio 3.0\n")
         fp.write( "short_energy_error_threshold 0.00\n")
         fp.write( "short_force_error_threshold 1.00\n")
         fp.write( "rmse_threshold_trials 3\n")
         fp.write( "weights_min -1.0\n")
         fp.write( "weights_max  1.0\n")
         fp.write( "main_error_metric RMSEpa\n")
         fp.write( "write_trainpoints 1\n")
         fp.write( "write_trainforces   1\n")
         fp.write( "write_weights_epoch 1\n")
         fp.write( "write_neuronstats   1\n")
         fp.write( "write_trainlog\n")
         fp.write( "kalman_type    0\n")
         fp.write( "kalman_epsilon 1.0E-2\n")
         fp.write( "kalman_q0      0.01\n")
         fp.write( "kalman_qtau    2.302\n")
         fp.write( "kalman_qmin    1.0E-6\n")
         fp.write( "kalman_eta     0.01\n")
         fp.write( "kalman_etatau  2.302\n")
         fp.write( "kalman_etamax  1.0\n")
     gen_symfunc(elements, Path("input.nn"), r_cutoff = 6.0)    

def append_random_seed(num: int) -> None:
    '''Append a random random seed to input.nn

    Everytime we call this function we create a new random number generator.
    This generator will be seeded from the hardware entropy pool (if available
    in your machine). Just incase there is no entropy pool we loop a number
    of times over the generator and draw a random number that we'll use as
    a seed. By ensuring we set num to a different value on every call we can
    still draw a unique seed.
    '''
    if num < 1:
        raise RuntimeError(f"append_random_seed: num must be positive: {str(num)}")
    random = np.random.default_rng()
    ival = random.integers(low=sys.maxsize)
    for ii in range(num):
       ival = random.integers(low=sys.maxsize)
    with open("input.nn","a") as fp:
        fp.write(f"random_seed {str(ival)}\n")

def create_directories(data_path: Path = None) -> None:
    '''Generate the directories for scaling and training runs
    '''
    #
    # Make directories if needed
    #
    os.makedirs("scaling",exist_ok=True)
    os.makedirs("train-1",exist_ok=True)
    os.makedirs("train-2",exist_ok=True)
    os.makedirs("train-3",exist_ok=True)
    os.makedirs("train-4",exist_ok=True)
    #
    # Softlink the training data
    #
    if data_path is None:
        data_path = Path("..") / "ab-initio" / "input.data"
    path = Path("scaling") / "input.data"
    if not path.exists():
        subprocess.run(["ln","-s",str(data_path),str(path)])
    path = Path("train-1") / "input.data"
    if not path.exists():
        subprocess.run(["ln","-s",str(data_path),str(path)])
    path = Path("train-2") / "input.data"
    if not path.exists():
        subprocess.run(["ln","-s",str(data_path),str(path)])
    path = Path("train-3") / "input.data"
    if not path.exists():
        subprocess.run(["ln","-s",str(data_path),str(path)])
    path = Path("train-4") / "input.data"
    if not path.exists():
        subprocess.run(["ln","-s",str(data_path),str(path)])
    #
    # Create input files
    #
    elements = read_elements(data_path)
    for ii in range(len(elements)):
        element = elements[ii]
        elements[ii] = element.capitalize()
    path = Path("scaling") / "input.nn"
    if not path.exists():
        os.chdir("scaling")
        write_input(elements,6,6.0)
        append_random_seed(1)
        os.chdir("..")
    path = Path("train-1") / "input.nn"
    if not path.exists():
        os.chdir("train-1")
        write_input(elements,6,6.0)
        append_random_seed(2)
        os.chdir("..")
    path = Path("train-2") / "input.nn"
    if not path.exists():
        os.chdir("train-2")
        write_input(elements,6,6.0)
        append_random_seed(3)
        os.chdir("..")
    path = Path("train-3") / "input.nn"
    if not path.exists():
        os.chdir("train-3")
        write_input(elements,6,6.0)
        append_random_seed(4)
        os.chdir("..")
    path = Path("train-4") / "input.nn"
    if not path.exists():
        os.chdir("train-4")
        write_input(elements,6,6.0)
        append_random_seed(5)
        os.chdir("..")
    #
    # Create softlinks to scaling.data (this file will be generated when nnp-scaling is run)
    #
    path = Path("train-1") / "scaling.data"
    if not path.exists():
        os.chdir("train-1")
        subprocess.run(["ln","-s","../scaling/scaling.data","scaling.data"])
        os.chdir("..")
    path = Path("train-2") / "scaling.data"
    if not path.exists():
        os.chdir("train-2")
        subprocess.run(["ln","-s","../scaling/scaling.data","scaling.data"])
        os.chdir("..")
    path = Path("train-3") / "scaling.data"
    if not path.exists():
        os.chdir("train-3")
        subprocess.run(["ln","-s","../scaling/scaling.data","scaling.data"])
        os.chdir("..")
    path = Path("train-4") / "scaling.data"
    if not path.exists():
        os.chdir("train-4")
        subprocess.run(["ln","-s","../scaling/scaling.data","scaling.data"])
        os.chdir("..")
