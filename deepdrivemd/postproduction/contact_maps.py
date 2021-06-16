import glob
import adios2

pattern = "/usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/1/aggregation_runs/stage0000/task*/agg.bp"

bpfiles = glob.glob(pattern)


for bp in bpfiles:
    with adios2.open(bp, "r") as fh:
        for fstep in fh:
            physical_time = fstep.read("contact_map")

