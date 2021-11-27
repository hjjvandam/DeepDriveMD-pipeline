import random
import numpy as np
import glob
import subprocess
import time
import sys
import os
import argparse
import itertools
from typing import List, Tuple
from numba import cuda
from pathlib import Path
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# from multiprocessing import Process, Pool
from pathos.multiprocessing import ProcessingPool as Pool

# import math

from deepdrivemd.utils import Timer, timer, t1Dto2D
from deepdrivemd.agents.stream.config import OutlierDetectionConfig
import tensorflow.keras.backend as K
from deepdrivemd.data.stream.enumerations import DataStructure

import pickle
from deepdrivemd.data.stream.OutlierDB import OutlierDB
from lockfile import LockFile
from deepdrivemd.data.stream.aggregator_reader import (
    Streams,
    StreamVariable,
    StreamContactMapVariable,
    StreamScalarVariable,
)

import cupy as cp
from cuml import DBSCAN as DBSCAN

from deepdrivemd.models.keras_cvae.model import CVAE
from simtk.openmm.app.pdbfile import PDBFile

import adios2

pool = Pool(39)


def clear_gpu():
    K.clear_session()
    try:
        device = int(os.environ["CUDA_VISIBLE_DEVICES"])
        print("device = ", device)
        cuda.select_device(device)
        cuda.close()
        sys.stdout.flush()
    except Exception as e:
        print(e)


def build_model(cfg: OutlierDetectionConfig, model_path: str):
    cvae = CVAE(
        image_size=cfg.final_shape,
        channels=cfg.final_shape[-1],
        conv_layers=cfg.conv_layers,
        feature_maps=cfg.conv_filters,
        filter_shapes=cfg.conv_filter_shapes,
        strides=cfg.conv_strides,
        dense_layers=cfg.dense_layers,
        dense_neurons=cfg.dense_neurons,
        dense_dropouts=cfg.dense_dropouts,
        latent_dim=cfg.latent_dim,
    )
    cvae.load(model_path)
    return cvae


def wait_for_model(cfg: OutlierDetectionConfig) -> str:
    """Wait for the trained model to be published by machine learning pipeline.

    Returns
    -------
    str
        Path to the model.
    """

    while True:
        if os.path.exists(cfg.best_model):
            break
        print(f"No model {cfg.best_model}, sleeping")
        sys.stdout.flush()
        time.sleep(cfg.timeout2)
    return cfg.best_model


def wait_for_input(cfg: OutlierDetectionConfig) -> List[str]:
    """Wait for enough data to be produced by simulations.

    Returns
    -------
    List[str]
        List of aggregated bp files.
    """
    while True:
        bpfiles = glob.glob(str(cfg.agg_dir / "*/*/agg.bp"))
        if len(bpfiles) == cfg.num_agg:
            break
        print(f"Waiting for {cfg.num_agg} agg.bp files")
        time.sleep(cfg.timeout1)

    print(f"bpfiles = {bpfiles}")

    # Wait for enough time steps in each bp file
    while True:
        enough = True
        for bp in bpfiles:
            com = f"bpls {bp}"
            a = subprocess.getstatusoutput(com)
            if a[0] != 0:
                enough = False
                print(f"Waiting, a = {a}, {bp}")
                break
            try:
                steps = int(a[1].split("\n")[0].split("*")[0].split(" ")[-1])
            except Exception as e:
                print("Exception ", e)
                steps = 0
                enough = False
            if steps < cfg.min_step_increment:
                enough = False
                print(f"Waiting, steps = {steps}, {bp}")
                break
        if enough:
            break
        else:
            time.sleep(cfg.timeout2)

    return bpfiles


def dirs(cfg: OutlierDetectionConfig) -> Tuple[str, str, str]:
    """Create tmp_dir and published_dir into which outliers are written
    Returns
    -------
    Tuple[str, str, str]
        Paths to temporary and published directories. As the outliers are found, they are first written into `tmp_dir` and late moved to `published_dir` from where they are taken by the simulations.
    """
    tmp_dir = cfg.output_path / "tmp"
    published_dir = cfg.output_path / "published_outliers"
    tmp_dir.mkdir(exist_ok=True)
    published_dir.mkdir(exist_ok=True)
    return tmp_dir, published_dir


def predict(
    cfg: OutlierDetectionConfig,
    model_path: str,
    cvae_input: Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ],
    batch_size: int = 32,
) -> np.ndarray:
    """Project contact maps into the middle layer of CVAE

    Parameters
    ----------
    cfg : OutlierDetectionConfig
    model_path : str
        Path to the published model.
    cvae_input : Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]
        Each list corresponds to the variable described by `variable_list` in `main` function. In particular, the first one contains contact maps and only those are used in `predict()`.
    batch_size : int
        Batch size used to project input to the middle layer of the autoencoder.
    Returns
    -------
    np.ndarray
        The latent space representation of the input.
    """
    input = np.expand_dims(cvae_input[0], axis=-1)

    cfg.initial_shape = input.shape[1:3]
    cfg.final_shape = list(input.shape[1:3]) + list(np.array([1]))

    cvae = build_model(cfg, model_path)

    cm_predict = cvae.return_embeddings(input, batch_size)
    del cvae
    clear_gpu()
    return cm_predict


def outliers_from_latent(
    cm_predict: np.ndarray, eps: float = 0.35, min_samples: int = 10
) -> np.ndarray:
    """Cluster the elements in the middle layer of CVAE.

    Parameters
    ----------
    cm_predict : np.ndarray[np.float32]
        Projections of contact maps to the middle layer of CVAE.
    eps : float
        DBSCAN's eps
    min_samples : int
        DBSCAN's min_samples.

    Returns
    -------
    np.ndarray
        Indices of outliers.
    """
    cm_predict = cp.asarray(cm_predict)
    db = DBSCAN(eps=eps, min_samples=min_samples, max_mbytes_per_batch=100).fit(
        cm_predict
    )
    db_label = db.labels_.to_array()
    print("unique labels = ", np.unique(db_label))
    outlier_list = np.where(db_label == -1)
    clear_gpu()
    return outlier_list


def cluster(
    cfg: OutlierDetectionConfig,
    cm_predict: np.ndarray,
    outlier_list: np.ndarray,
    eps: float,
    min_samples: int,
) -> Tuple[float, int]:
    """Run :obj:`outliers_from_latent` changing parameters of DBSCAN until
    the desired number of outliers is obtained.

    Parameters
    ----------
    cfg : OutlierDetectionConfig
    cm_predict : np.ndarray
    outlier_list : np.ndaray
    eps : float
    min_samples : int

    Returns
    -------
    Tuple[float, int]
        eps, min_samples which give the number of outliers in the desired range.
    """
    outlier_count = cfg.outlier_count
    while outlier_count > 0:
        n_outlier = 0
        try:
            outliers = np.squeeze(
                outliers_from_latent(cm_predict, eps=eps, min_samples=min_samples)
            )
            n_outlier = len(outliers)
        except Exception as e:
            print(e)
            print("No outliers found")

        print(
            f"eps = {eps}, min_samples = {min_samples}, number of outlier found: {n_outlier}"
        )

        if n_outlier > cfg.outlier_max:
            eps = eps + 0.09 * random.random()
            min_samples -= int(random.random() < 0.5)
            min_samples = max(5, min_samples)
        elif n_outlier < cfg.outlier_min:
            eps = max(0.01, eps - 0.09 * random.random())
            min_samples += int(random.random() < 0.5)
        else:
            outlier_list.append(outliers)
            break
        outlier_count -= 1
    return eps, min_samples


def write_pdb_frame(
    frame: np.ndarray, original_pdb: Path, output_pdb_fn: str, ligand: int
):
    """Write positions into pdb file.

    Parameters
    ----------
    frame : np.ndarray
        Positions of atoms.
    original_pdb : str
        PDB file with initial condition to be used for topology.
    output_pdb_fn : str
        Where to write an outlier.
    """
    pdb = PDBFile(str(original_pdb))
    print(
        "write_pdb_frame:  original_pdb = ",
        original_pdb,
        " frame.shape = ",
        frame.shape,
        " ligand = ",
        ligand,
    )
    sys.stdout.flush()
    with open(str(output_pdb_fn), "w") as f:
        try:
            PDBFile.writeFile(pdb.getTopology(), frame, f)
        except Exception as e:
            print(
                e,
                "\n",
                "original_pdb = ",
                str(original_pdb),
                "frame.shape = ",
                frame.shape,
                "output_pdb_fn = ",
                str(output_pdb_fn),
                "ligand = ",
                ligand,
                file=sys.stderr,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            raise e
        f.flush()
        f.flush()

    del pdb

    sys.stdout.flush()
    sys.stderr.flush()

    """
    print('output_pdb_fn: ', output_pdb_fn, subprocess.getstatusoutput(f"ls -l {output_pdb_fn}"),
          subprocess.getstatusoutput(f"md5sum {output_pdb_fn}"),
          subprocess.getstatusoutput("free -h"), ', parent process: ', os.getppid(),
          ', process id: ', os.getpid(), ', core: ',
          open("/proc/{pid}/stat".format(pid=os.getpid()), 'rb').read().split()[-14] )
    sys.stdout.flush()
    """


def check_output(dir):
    print("=" * 30)
    print(subprocess.getstatusoutput(f"ls -l {dir}/*")[1])
    print("=" * 30)
    print(subprocess.getstatusoutput(f"md5sum {dir}/*")[1])
    print("=" * 30)
    sys.stdout.flush()


def write_top_outliers(
    cfg: OutlierDetectionConfig,
    tmp_dir: str,
    top: Tuple[np.ndarray, np.ndarray, np.ndarray],
):
    """Save to PDB files top outliers.

    Parameters
    ----------
    cfg : OutlierDetectionConfig
    tmp_dir : str
          Temporary directory to write outliers to.
    top : Tuple[np.ndarray, np.ndarray, np.ndarray]
          Top :obj:`N` positions, velocities, md5sums where
          :obj:`N` is equal to the number of the simulations.
    """
    positions, velocities, md5s = top[:3]

    pp = []

    if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
        dirs = top[5]
        print("dirs=")
        print(dirs)
        table = pd.read_csv(cfg.multi_ligand_table)
        for p, v, m, d in zip(positions, velocities, md5s, dirs):
            print("d=", d)
            sys.stdout.flush()
            d = int(d)
            print("   d=", d)
            topology_file = table["pdb"][d]
            tdir = table["tdir"][d]
            outlier_pdb_file = f"{tmp_dir}/{m}.pdb"
            outlier_v_file = f"{tmp_dir}/{m}.npy"
            init_pdb_file = Path(f"{tdir}/system/{topology_file}")

            pp.append(
                pool.apipe(
                    write_pdb_frame, p.copy(), init_pdb_file, outlier_pdb_file, d
                )
            )
            pp.append(pool.apipe(np.save, outlier_v_file, v))
            """
            write_pdb_frame(p, init_pdb_file, outlier_pdb_file)
            np.save(outlier_v_file, v)
            """
            task_file = f"{tmp_dir}/{m}.txt"
            with open(task_file, "w") as f:
                f.write(str(d))
                f.flush()
    else:
        for p, v, m in zip(positions, velocities, md5s):
            outlier_pdb_file = f"{tmp_dir}/{m}.pdb"
            outlier_v_file = f"{tmp_dir}/{m}.npy"
            pp.append(
                pool.apipe(write_pdb_frame, p, cfg.init_pdb_file, outlier_pdb_file, -1)
            )
            pp.append(pool.apipe(np.save, outlier_v_file, v))

    for p in pp:
        zz = p.get()
        print(zz)

    sys.stdout.flush()
    sys.stderr.flush()

    check_output(tmp_dir)


def write_db(top: Path, tmp_dir: Path) -> OutlierDB:
    """Create and save a database of outliers to be used by simulation."""
    outlier_db_fn = f"{tmp_dir}/OutlierDB.pickle"
    outlier_files = list(map(lambda x: f"{tmp_dir}/{x}.pdb", top[2]))
    rmsds = top[3]
    db = OutlierDB(tmp_dir, list(zip(rmsds, outlier_files)))
    with open(outlier_db_fn, "wb") as f:
        pickle.dump(db, f)
    return db


def publish(tmp_dir: Path, published_dir: Path):
    """Publish outliers and the corresponding database for simulations to pick up."""
    dbfn = f"{published_dir}/OutlierDB.pickle"
    subprocess.getstatusoutput(f"touch {dbfn}")

    mylock = LockFile(dbfn)

    mylock.acquire()
    print(
        subprocess.getstatusoutput(
            f"rm -rf {published_dir}/*.pickle {published_dir}/*.pdb {published_dir}/*.npy"
        )
    )
    print(subprocess.getstatusoutput(f"mv {tmp_dir}/* {published_dir}/"))
    mylock.release()


def top_outliers(
    cfg: OutlierDetectionConfig,
    cvae_input: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    outlier_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find top :obj:num_sim` outliers sorted by :obj:`rmsd`.

    Parameters
    ----------
    cfg : OutlierDetectionConfig
    cvae_input : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            steps, positions, velocities, md5sums, rmsds
    outlier_list : np.ndarray
            indices corresponding to outliers

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
         Positions, velocities, md5sums, rmsds, outlier
         indices of outliers, sorted in ascending order by rmsd
    """
    outlier_list = list(outlier_list[0])
    positions = cvae_input[1][outlier_list]
    velocities = cvae_input[3][outlier_list]
    md5s = cvae_input[2][outlier_list]
    rmsds = cvae_input[4][outlier_list]

    z = list(zip(positions, velocities, md5s, rmsds, outlier_list))
    z.sort(key=lambda x: x[3])
    z = z[: cfg.num_sim]
    z = list(zip(*z))

    return z


def random_outliers(
    cfg: OutlierDetectionConfig,
    cvae_input: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    outlier_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find :obj:`num_sim` outliers in a random order. Can be used in the absense of :obj:`rmsd`.

    Parameters
    ----------
    cfg : OutlierDetectionConfig
    cvae_input : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            steps, positions, velocities, md5sums, rmsds
    outlier_list : np.ndarray
            indices corresponding to outliers

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
         Positions, velocities, md5sums, rmsds, outlier
         indices of outliers in a random order.
    """
    outlier_list = list(outlier_list[0])
    positions = cvae_input[1][outlier_list]
    velocities = cvae_input[3][outlier_list]
    md5s = cvae_input[2][outlier_list]
    if cfg.compute_rmsd:
        rmsds = cvae_input[4][outlier_list]
    else:
        rmsds = np.array([-1.0] * len(outlier_list))

    if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
        dirs = cvae_input[-1][outlier_list]
        z = list(zip(positions, velocities, md5s, rmsds, outlier_list, dirs))
    else:
        z = list(zip(positions, velocities, md5s, rmsds, outlier_list))
    indices = np.arange(len(z))
    np.random.shuffle(indices)
    indices = indices[: cfg.num_sim]
    z = [z[i] for i in indices]
    z = list(zip(*z))

    return z


def run_lof(data: np.ndarray) -> np.ndarray:
    clf = LocalOutlierFactor()
    clf.fit_predict(data)
    lof_scores = clf.negative_outlier_factor_
    return lof_scores


def top_lof(
    cfg: OutlierDetectionConfig,
    cvae_input: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cm_predict: np.array,
    outlier_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    outlier_list = list(outlier_list[0])
    projections = cm_predict[outlier_list]
    lof_scores = run_lof(projections)
    print("lof_scores = ", lof_scores)
    sys.stdout.flush()
    positions = cvae_input[1][outlier_list]
    velocities = cvae_input[3][outlier_list]
    md5s = cvae_input[2][outlier_list]

    if cfg.compute_rmsd:
        rmsds = cvae_input[4][outlier_list]
    else:
        rmsds = np.array([-1.0] * len(outlier_list))

    if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
        dirs = cvae_input[-1][outlier_list]
        z = list(
            zip(positions, velocities, md5s, rmsds, outlier_list, dirs, lof_scores)
        )
        z.sort(key=lambda x: x[6])
    else:
        z = list(zip(positions, velocities, md5s, rmsds, outlier_list, lof_scores))
        z.sort(key=lambda x: x[5])

    z = z[: cfg.num_sim]
    z = list(zip(*z))
    return z


def select_best_random(
    cfg: OutlierDetectionConfig,
    cvae_input: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> List[int]:
    """Sort cvae_input by rmsd, selects :obj:`2*cfg.num_sim` best entries, out of them
    randomly select :obj:`cfg.num_sim`, return the corresponding indices.

    Parameters
    ----------
    cfg : OutlierDetectionConfig
    cvae_input : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        steps, positions, velocities, md5sums, rmsds

    Returns
    -------
    List[int]
        List of :obj:`cfg.num_sim` indices randomly selected from a list of
        :obj:`2*cfg.num_sim` entries with smallest rmsd.

    Note
    ----
    This is used when no outliers are found.
    """
    rmsds = cvae_input[4]
    z = sorted(zip(rmsds, range(len(rmsds))), key=lambda x: x[0])
    sorted_index = list(map(lambda x: x[1], z))[2 * cfg.num_sim :]
    sorted_index = random.sample(sorted_index, cfg.num_sim)
    return sorted_index


def select_best(
    cfg: OutlierDetectionConfig,
    cvae_input: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> List[int]:
    """Sort cvae_input by rmsd, selects best :obj:`cfg.num_sim`, return
    the corresponding indices.

    Parameters
    ----------
    cfg : OutlierDetectionConfig
    cvae_input : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        steps, positions, velocities, md5sums, rmsds

    Returns
    -------
    List[int]
        List of :obj:`cfg.num_sim` indices for best traversed states among
        :obj:`lastN` from each aggregator.
    """
    rmsds = cvae_input[4]
    z = sorted(zip(rmsds, range(len(rmsds))), key=lambda x: x[0])
    sorted_index = list(map(lambda x: x[1], z))[cfg.num_sim :]
    return sorted_index


def main(cfg: OutlierDetectionConfig):
    print(subprocess.getstatusoutput("hostname")[1])
    sys.stdout.flush()

    print(cfg)

    with Timer("wait_for_input"):
        adios_files_list = wait_for_input(cfg)

    variable_list = [
        StreamContactMapVariable("contact_map", np.uint8, DataStructure.array),
        StreamVariable("positions", np.float32, DataStructure.array),
        StreamVariable("md5", str, DataStructure.string),
        StreamVariable("velocities", np.float32, DataStructure.array),
    ]

    if cfg.compute_rmsd:
        variable_list.append(
            StreamScalarVariable("rmsd", np.float32, DataStructure.scalar)
        )

    if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
        variable_list.append(StreamVariable("ligand", np.int32, DataStructure.scalar))

    mystreams = Streams(
        adios_files_list,
        variable_list,
        lastN=cfg.lastN,
        config=cfg.adios_xml_agg,
        stream_name="AggregatorOutput",
        batch=cfg.read_batch,
    )

    with Timer("outlier_read"):
        cvae_input = mystreams.next()

    with Timer("wait_for_model"):
        model_path = str(wait_for_model(cfg))

    tmp_dir, published_dir = dirs(cfg)
    eps = cfg.init_eps
    min_samples = cfg.init_min_samples

    # Infinite loop of outlier search iterations
    for j in itertools.count(0):
        print(f"outlier iteration {j}")

        timer("outlier_search_iteration", 1)

        """
        with Timer("outlier_read"):
            cvae_input = mystreams.next()
            print("len(cvae_input) = ", len(cvae_input))
            print("type(cvae_input[-1]) = ", type(cvae_input[-1]))
            sys.stdout.flush()
            print("cvae_input[-1].shape = ", cvae_input[-1].shape)
            sys.stdout.flush()
        """

        with Timer("outlier_predict"):
            cm_predict = predict(cfg, model_path, cvae_input)

        outlier_list = []
        with Timer("outlier_cluster"):
            eps, min_samples = cluster(cfg, cm_predict, outlier_list, eps, min_samples)
            if (
                cfg.use_outliers is False
                or len(outlier_list) == 0
                or len(outlier_list[0]) < cfg.num_sim
            ):
                print("Not using outliers")
                clear_gpu()
                if cfg.compute_rmsd:
                    print("Using best rmsd states")
                    outlier_list = [select_best(cfg, cvae_input)]
                else:
                    print("Using random states")
                    outlier_list = [
                        list(
                            np.random.choice(
                                np.arange(len(cvae_input[0])),
                                cfg.num_sim,
                                replace=False,
                            )
                        )
                    ]
                eps = cfg.init_eps
                min_samples = cfg.init_min_samples
        if cfg.outlier_selection == "lof":
            print("Using top lof outliers")
            top = top_lof(cfg, cvae_input, cm_predict, outlier_list)
        elif cfg.use_random_outliers or (not cfg.compute_rmsd):
            print("Using random outliers")
            top = random_outliers(cfg, cvae_input, outlier_list)
        else:
            print("Using top outliers sorted by rmsd")
            top = top_outliers(cfg, cvae_input, outlier_list)

        print("top outliers = ", top[3])

        with Timer("outlier_write"):
            write_top_outliers(cfg, tmp_dir, top)

        with Timer("outlier_db"):
            write_db(top, tmp_dir)

        with Timer("outlier_publish"):
            publish(tmp_dir, published_dir)

        with Timer("outlier_read"):
            cvae_input = mystreams.next()

        timer("outlier_search_iteration", -1)


def read_lastN(
    adios_files_list: List[str], lastN: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Read :obj:`lastN` steps from each aggregated file. Used by :obj:`project()`

    Parameters
    ----------
    adios_files_list : List[str]
        A list of aggregated adios files.
    lastN :int
        How many last entries to get from each file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        :obj:`lastN` contact maps from each aggregated file and
        :obj:`lastN` corresponding rmsds.
    """

    vars = ["contact_map"]

    if hasattr(cfg, "compute_zcentroid") and cfg.compute_zcentroid:
        print("compute_zcentroid = ", cfg.compute_zcentroid)
        sys.stdout.flush()
        vars.append("zcentroid")

    if cfg.compute_rmsd:
        vars.append("rmsd")

    if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
        vars.append("ligand")
        vars.append("dir")

    variable_lists = {}
    for bp in adios_files_list:
        with adios2.open(bp, "r") as fh:
            steps = fh.steps()
            start_step = steps - lastN - 2
            if start_step < 0:
                start_step = 0
                lastN = steps
            for v in vars:
                if v == "contact_map":
                    shape = list(
                        map(int, fh.available_variables()[v]["Shape"].split(","))
                    )
                elif v == "rmsd" or v == "zcentroid":
                    print(fh.available_variables()[v]["Shape"])
                    sys.stdout.flush()
                if v == "contact_map":
                    start = [0] * len(shape)
                    var = fh.read(
                        v,
                        start=start,
                        count=shape,
                        step_start=start_step,
                        step_count=lastN,
                    )
                elif v == "rmsd" or v == "zcentroid" or v == "ligand":
                    var = fh.read(v, [], [], step_start=start_step, step_count=lastN)
                elif v == "dir":
                    var = fh.read_string(v, step_start=start_step, step_count=lastN)
                if v != "dir":
                    print("v = ", v, " var.shape = ", var.shape)
                else:
                    print("v = ", v, " len(var) = ", len(var))
                try:
                    variable_lists[v].append(var)
                except Exception as e:
                    print("Exception ", e)
                    variable_lists[v] = [var]

    for vl in variable_lists:
        print(vl)
        print(len(variable_lists[vl]))
        variable_lists[vl] = np.vstack(variable_lists[vl])
        print(len(variable_lists[vl]))

    variable_lists["contact_map"] = np.array(
        list(
            map(
                lambda x: t1Dto2D(np.unpackbits(x.astype("uint8"))),
                list(variable_lists["contact_map"]),
            )
        )
    )

    print(variable_lists["contact_map"].shape)
    if cfg.compute_rmsd:
        print(variable_lists["rmsd"].shape)
    sys.stdout.flush()

    result = [variable_lists["contact_map"]]

    if hasattr(cfg, "compute_zcentroid") and cfg.compute_zcentroid:
        result.append(np.concatenate(variable_lists["zcentroid"]))

    if cfg.compute_rmsd:
        result.append(np.concatenate(variable_lists["rmsd"]))

    if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
        result.append(np.concatenate(variable_lists["ligand"]))
        result.append(np.concatenate(variable_lists["dir"]))

    return tuple(result)


def project_mini(cfg: OutlierDetectionConfig):
    with Timer("wait_for_input"):
        adios_files_list = wait_for_input(cfg)
    with Timer("wait_for_model"):
        model_path = str(wait_for_model(cfg))
        print("model_path = ", model_path)

    lastN = cfg.project_lastN

    # Create output directories
    dirs(cfg)

    for i, bp in enumerate(adios_files_list):
        print(f"i={i}, bp={bp}")
        sys.stdout.flush()
        cvae_input = read_lastN([bp], lastN)

        if hasattr(cfg, "compute_zcentroid") and cfg.compute_zcentroid:
            zcentroid = cvae_input[1]
            with open(cfg.output_path / f"zcentroid_{i}.npy", "wb") as f:
                np.save(f, zcentroid)

        if cfg.compute_rmsd:
            rmsds = cvae_input[1]  # cvae_input[2]
            with open(cfg.output_path / f"rmsd_{i}.npy", "wb") as f:
                np.save(f, rmsds)

        if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
            ligand = cvae_input[2]
            sim = cvae_input[3]
            for j in range(len(ligand)):
                print(f"ligand[{j}] = {ligand[j]}")
                if ligand[j] == -1:
                    ligand[j] = int(sim[j])
            with open(cfg.output_path / f"ligand_{i}.npy", "wb") as f:
                np.save(f, ligand)

        with Timer("project_predict"):
            embeddings_cvae = predict(cfg, model_path, cvae_input, batch_size=64)
        with open(cfg.output_path / f"embeddings_cvae_{i}.npy", "wb") as f:
            np.save(f, embeddings_cvae)


def project(cfg: OutlierDetectionConfig):
    """Postproduction: compute t-SNE embeddings."""
    if cfg.project_gpu:
        from cuml import TSNE
    else:
        from sklearn.manifold import TSNE

    with Timer("wait_for_input"):
        adios_files_list = wait_for_input(cfg)
    with Timer("wait_for_model"):
        model_path = str(wait_for_model(cfg))
        print("model_path = ", model_path)

    lastN = cfg.project_lastN

    # Create output directories
    dirs(cfg)

    with Timer("project_next"):
        cvae_input = read_lastN(adios_files_list, lastN)

    if cfg.compute_rmsd:
        rmsds = cvae_input[1]
        with open(cfg.output_path / "rmsd.npy", "wb") as f:
            np.save(f, rmsds)

    with Timer("project_predict"):
        embeddings_cvae = predict(cfg, model_path, cvae_input, batch_size=1024)

    with open(cfg.output_path / "embeddings_cvae.npy", "wb") as f:
        np.save(f, embeddings_cvae)

    with Timer("project_TSNE_2D"):
        tsne2 = TSNE(n_components=2)
        tsne_embeddings2 = tsne2.fit_transform(embeddings_cvae)

    with open(cfg.output_path / "tsne_embeddings_2.npy", "wb") as f:
        np.save(f, tsne_embeddings2)

    with Timer("project_TSNE_3D"):
        tsne3 = TSNE(n_components=3)
        tsne_embeddings3 = tsne3.fit_transform(embeddings_cvae)

    with open(cfg.output_path / "tsne_embeddings_3.npy", "wb") as f:
        np.save(f, tsne_embeddings3)


def project_tsne_3D(cfg: OutlierDetectionConfig):
    from sklearn.manifold import TSNE

    tsne3 = TSNE(n_components=3)
    emb = []
    for i in range(10):
        with open(cfg.output_path / f"embeddings_cvae_{i}.npy", "rb") as f:
            emb.append(np.load(f))
    embeddings_cvae = np.concatenate(emb)

    with Timer("project_TSNE_3D"):
        tsne_embeddings3 = tsne3.fit_transform(embeddings_cvae)

    with open(cfg.output_path / "tsne_embeddings_3.npy", "wb") as f:
        np.save(f, tsne_embeddings3)


def project_tsne_2D(cfg: OutlierDetectionConfig):
    from sklearn.manifold import TSNE

    tsne2 = TSNE(n_components=2)
    emb = []
    rmsds = []
    ligands = []
    for i in range(9):
        with open(cfg.output_path / f"embeddings_cvae_{i}.npy", "rb") as f:
            emb.append(np.load(f))
        with open(cfg.output_path / f"rmsd_{i}.npy", "rb") as f:
            rmsds.append(np.load(f))
        with open(cfg.output_path / f"ligand_{i}.npy", "rb") as f:
            ligands.append(np.load(f))

    embeddings_cvae = np.concatenate(emb)
    RMSDS = np.concatenate(rmsds)
    with open(cfg.output_path / "rmsds.npy", "wb") as f:
        np.save(f, RMSDS)
    LIGANDS = np.concatenate(ligands)
    with open(cfg.output_path / "ligands.npy", "wb") as f:
        np.save(f, LIGANDS)
    with Timer("project_TSNE_2D"):
        tsne_embeddings2 = tsne2.fit_transform(embeddings_cvae)

    with open(cfg.output_path / "tsne_embeddings_2.npy", "wb") as f:
        np.save(f, tsne_embeddings2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    parser.add_argument("-p", "--project", action="store_true", help="compute tsne")
    parser.add_argument(
        "-m", "--miniproject", action="store_true", help="compute embeddings only"
    )
    parser.add_argument(
        "-T",
        "--tsne_2D",
        action="store_true",
        help="compute 2D tsne, assuming embeddings are already computed",
    )
    parser.add_argument(
        "-t",
        "--tsne_3D",
        action="store_true",
        help="compute 3D tsne, assuming embeddings are already computed",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = OutlierDetectionConfig.from_yaml(args.config)

    if args.project:
        project(cfg)
    elif args.miniproject:
        project_mini(cfg)
    elif args.tsne_3D:
        project_tsne_3D(cfg)
    elif args.tsne_2D:
        project_tsne_2D(cfg)
    else:
        main(cfg)
