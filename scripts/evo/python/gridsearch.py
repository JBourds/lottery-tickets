import itertools
import os
import subprocess

SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 
    "shell",
    "base.sh",
)

mutation_scales = [.1, .25, .5]
mutation_rates = [.125, .25, .5]
layers_list = [
    [],
    [(4, "relu")],
    [(8, "relu")],
    [(4, "relu"), (4, "relu")],
]

anames = ["lenet"]
dnames = ["mnist"]
asizes = [10]
tsizes = [4]
nwinners_list = [2]
psizes = [25]
nfronts_list = [2]
experiments_list = [10]
generations_list = [100]

for aname, dname, mrate, mscale, layers, asize, tsize, nwinners, psize, nfronts, experiments, generations in itertools.product(
    anames,
    dnames,
    mutation_rates,
    mutation_scales,
    layers_list,
    asizes,
    tsizes,
    nwinners_list,
    psizes,
    nfronts_list,
    experiments_list,
    generations_list,
):
    hidden_sizes = ",".join(map(lambda tup: str(tup[0]), layers))
    if not hidden_sizes:
        hidden_sizes = "None"
    hidden_activations = ",".join(map(lambda tup: tup[1], layers))
    if not hidden_activations:
        hidden_activations = "None"
    aname, dname, mrate, mscale, asize, tsize, nwinners, psize, nfronts, experiments, generations = map(
        str,
        (aname, dname, mrate, mscale, asize, tsize, nwinners, psize, nfronts, experiments, generations)
    )
   
    cmd = f"nohup {SCRIPT_PATH} " \
        + f"--aname={aname} " \
        + f"--dname={dname} " \
        + f"--mrate={mrate} " \
        + f"--mscale={mscale} " \
        + f"--hidden_sizes={hidden_sizes} " \
        + f"--hidden_activations={hidden_activations} " \
        + f"--asize={asize} " \
        + f"--tsize={tsize} " \
        + f"--nwinners={nwinners} " \
        + f"--psize={psize} " \
        + f"--nfronts={nfronts} " \
        + f"--experiments={experiments} " \
        + f"--generations={generations} " \
        + f"--vacc " \
        + f"{0} > /dev/null 2>&1 &"
    print(cmd)
    os.system(cmd)
   
