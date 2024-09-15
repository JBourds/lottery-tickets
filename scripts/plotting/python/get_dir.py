
import argparse
import datetime
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for getting the directory an experiment is run in.')

    # Experiment params
    parser.add_argument('--dir', type=str, default=None,
                        help='Output directory to store all models and experiment summary.')

    args, unknown = parser.parse_known_args()

    # Construct the full experiment directory path
    print(args.dir)
