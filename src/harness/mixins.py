"""
mixing.py

Module containing mixing classes which provide various utilities.

Author: Jordan Bourdeau
Date Created: 4/28/24
"""

import pickle

class PickleMixin:
    def save_to(self, filepath: str):
        """
        Save the object to a file using pickle.

        Args:
            filepath (str): The path to save the object.
        """
        with open(filepath, 'wb') as file:
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