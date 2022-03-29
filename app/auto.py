from subprocess import run
from time import sleep

parts = ["2000", "5000"]
rtypes = ["cpu_blr", "mgpu_blr", "parla"]
sizes = ["20k"]


#sizes = ["10k", "20k"]
#parts = ["50"] 
#sizes = ["1k"]

parla_gpus = ["0", "0,1", None]


template_cmd = "python app/main.py run {rtype} inputs/Arand{size}.mat.npy inputs/xrand{size}.mat.npy {psize} {lazy_or_eager} {manual_or_sched}"

for p in parts:
    for rt in rtypes:
        for isize in sizes:
            
            ngpus = [None]
            if rt == "parla":
                ngpus = parla_gpus
            for pgpu in ngpus:

                fname = f"{rt}_part{p}_inp{isize}"
                cmd = template_cmd.format(rtype=rt, size=isize, psize=p, fout=fname)

                if rt == "parla":
                    prefix = ""
                    if pgpu is not None:
                        prefix = f"CUDA_VISIBLE_DEVICES={pgpu} "
                    cmd = prefix + cmd

                    fprefix = "_4gpu"
                    if pgpu == "0":
                        fprefix = "_1gpu"
                    elif pgpu == "0,1":
                        fprefix = "_2gpu"

                    fname = fname+fprefix

                print("running  ", cmd)
                with open(fname+".txt", "w") as outfile:
                    run(cmd, shell=True, stdout=outfile)
                sleep(2)
