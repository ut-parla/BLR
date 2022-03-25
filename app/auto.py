from subprocess import run
from time import sleep

#parts = ["2000", "5000"]
parts = ["50"] 
rtypes = ["cpu_blr", "mgpu_blr", "parla"]
#sizes = ["10k", "20k"]
sizes = ["20k"]
sizes = ["1k"]


template_cmd = "python app/main.py run {rtype} inputs/Arand{size}.mat.npy inputs/xrand{size}.mat.npy {psize}"

for p in parts:
    for rt in rtypes:
        for isize in sizes:
            fname = f"{rt}_part{p}_inp{isize}.txt"
            cmd = template_cmd.format(rtype=rt, size=isize, psize=p, fout=fname)
            print("running  ", cmd)
            with open(fname, "w") as outfile:
                run(cmd, shell=True, stdout=outfile)
            sleep(2)
