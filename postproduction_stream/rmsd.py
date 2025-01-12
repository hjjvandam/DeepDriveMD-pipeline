import numpy as np
import adios2
import sys
import glob
import os

bpdir = sys.argv[1]
# /usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/5/aggregation_runs/stage0000/
outdir = sys.argv[2]


bpfiles = glob.glob(f"{bpdir}/*/agg.bp")
bpfiles.sort()

print("bpfiles = ", bpfiles)

for fn in bpfiles:
    taskid = int(os.path.basename(os.path.dirname(fn)).replace("task", ""))
    print(f"taskid = {taskid}")
    with adios2.open(fn, "r") as fr:
        n = fr.steps()
        rmsd = fr.read("rmsd", [], [], 0, n)
        print(
            f"n={n}, min(rmsd) = {np.min(rmsd)}, max(rms) = {np.max(rmsd)}, mean(rmsd) = {np.mean(rmsd)}, median(rmsd) = {np.median(rmsd)}"
        )
        np.save(f"{outdir}/rmsd_{taskid}.npy", rmsd)
        print("=" * 30)
        sys.stdout.flush()
