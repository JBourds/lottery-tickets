"""
mixing.py

Module containing mixing classes which provide various utilities.

Author: Jordan Bourdeau
Date Created: 4/28/24
"""

from datetime import datetime
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
        print("Loading")
        with open(filepath, 'rb') as file:
            return pickle.load(file)

class TimerMixin:

    def start_timer(self):
        """
        Start the timer.
        """
        self.start_time = datetime.now()

    def stop_timer(self):
        """
        Stop the timer.
        """
        self.end_time = datetime.now()

    def set_start_time(self, time: datetime):
        """
        Set the start time.

        Args:
            time (datetime): Datetime object for the start time.
        """
        self.start = time

    def set_end_time(self, time: datetime):
        """
        Set the end time.

        Args:
            time (datetime): Datetime object for the end time.
        """
        self.start = time

    def get_elapsed_time(self, units: str = 'seconds') -> float:
        """
        Get the elapsed time between start and end in the specified units.

        Args:
            units (str): Units in which to return the elapsed time. 
                         Possible values: 'seconds', 'minutes', 'hours', 'days'.

        Returns:
            float: Time elapsed between start and end in the specified units.
        """
        if self.start_time is None:
            raise ValueError("Timer has not been started.")

        if self.end_time is None:
            end_time = datetime.now()
        else:
            end_time = self.end_time

        elapsed = end_time - self.start_time

        if units == 'seconds':
            return elapsed.total_seconds()
        elif units == 'minutes':
            return elapsed.total_seconds() / 60
        elif units == 'hours':
            return elapsed.total_seconds() / 3600
        elif units == 'days':
            return elapsed.total_seconds() / 86400
        else:
            raise ValueError("Invalid units. Please choose from 'seconds', 'minutes', 'hours', 'days'.")
