import argparse
import datetime
import os

from src.harness import constants as C

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for creating the directory an experiment is run in.')

    # Logging
    parser.add_argument('--log_level', type=int, default=2,
                        help='Logging level to use. 0 = Not Set, 1 = Debug, 2 = Info, 3 = Warning, 4 = Error, 5 = Critical.')

    args, unknown = parser.parse_known_args()

    # Construct the full experiment directory path
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"evo_{args.model}_{args.dataset}_{args.experiments}_experiments_{timestamp}"
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
        C.EXPERIMENTS_DIRECTORY,
        path,
    )
    print(path)
