"""
mixing.py

Module containing mixing classes which provide various utilities.

Author: Jordan Bourdeau
Date Created: 4/28/24
"""

import os
import pickle

from src.harness import paths

class PickleMixin:
    def save_to(self, directory: str, filename: str):
        """
        Save the object to a file using pickle.

        Args:
            directory (str): The path to save the object.
            path (str): The file name to use
        """
        paths.create_path(directory)
        with open(os.path.join(directory, filename), 'wb') as file:
            pickle.dump(self, file)
    
    @classmethod
    def load_from(cls, filepath: str):
        """
        Load an object from a file using pickle.

        Args:
            filepath (str): The path from which to load the object.

        Returns:
            object: The loaded object.
        """
        with open(filepath, 'rb') as file:
            return pickle.load(file)