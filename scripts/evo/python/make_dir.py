import argparse
import datetime
import os

from src.harness import constants as C

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for creating the directory an experiment is run in.')

    # Hidden layers
    parser.add_argument("--hidden_sizes", type=str, default="", help="--hidden_sizes=[n,]")
    parser.add_argument("--hidden_activations", type=str, default="", help="--hidden_activations=[type,]")

    # Num experiments
    parser.add_argument("--experiments", type=int, default=1, help="--experiments=n")
    
    # Archive size
    parser.add_argument("--asize", type=int, default=10, help="--asize=n")

    # Population size
    parser.add_argument("--psize", type=int, default=10, help="--psize=n")

    # Fronts to consider
    parser.add_argument("--nfronts", type=int, default=2, help="--nfronts=n")

    # Tournament size
    parser.add_argument("--tsize", type=int, default=10, help="--psize=n")

    # Num tournament winners
    parser.add_argument("--nwinners", type=int, default=2, help="--nfronts=n")

    # Architecture and dataset names
    parser.add_argument("--aname", type=str, default="lenet", help="--aname=<arch>")
    parser.add_argument("--dname", type=str, default="mnist", help="--dname=<dataset>")

    # Mutation (hardcoded to simulated annealing, allow rate/scale tweaks)
    parser.add_argument("--mscale", type=float, default=0.5, help="--mscale=n")
    parser.add_argument("--mrate", type=float, default=0.2, help="--mrate=n")

    # Logging
    parser.add_argument('--log_level', type=int, default=2,
                        help='Logging level to use. 0 = Not Set, 1 = Debug, 2 = Info, 3 = Warning, 4 = Error, 5 = Critical.')

    args, unknown = parser.parse_known_args()

    # Construct the full experiment directory path
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"evo_{args.aname}_{args.dname}_{args.experiments}_experiments_{args.hidden_sizes}-{args.hidden_activations}" \
        + f"_layers_{args.asize}-asize_{args.psize}-psize_{args.nfronts}-nfronts_{args.tsize}-tsize_{args.nwinners}-nwinners" \
        + f"_{args.mscale}-mscale_{args.mrate}-mrate_{timestamp}"
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
        C.EXPERIMENTS_DIRECTORY,
        path,
    )
    print(path)
