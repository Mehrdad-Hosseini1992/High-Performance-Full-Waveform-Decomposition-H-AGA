# waveform.py
# Defines the Waveform class, a simple data structure to hold and provide
# easy access to the time-series data from the CSV files.

import pandas as pd
import numpy as np
from typing import List

class Waveform:
    """
    A data class to hold and manage waveform data from a single file.
    """
    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe.copy()
        # Check if 'Time' column exists, if not use the first column as Time
        if 'Time' not in self._df.columns and len(self._df.columns) >= 2:
            self._df.columns = ['Time'] + list(self._df.columns[1:])
        self._df.set_index('Time', inplace=True)

    def get_time_data(self) -> np.ndarray:
        """Returns the time data as a NumPy array."""
        return self._df.index.to_numpy(dtype=float)

    def get_data_channels(self) -> List[str]:
        """Returns a list of all data channel names (excluding Time)."""
        return self._df.columns.tolist()

    def has_data(self) -> bool:
        """Checks if the waveform has at least one data channel."""
        return len(self._df.columns) > 0

    def get_channel_names(self) -> List[str]:
        """Returns a list of all channel names."""
        return self._df.columns.tolist()

    def is_empty(self) -> bool:
        """Checks if the waveform DataFrame is empty."""
        return self._df.empty
    

    def get_channel_data(self, channel_name: str) -> np.ndarray:
        """
        Returns the amplitude data for a specific channel.
        
        Args:
            channel_name (str): The name of the channel to retrieve.
        
        Returns:
            np.ndarray: The amplitude data as a NumPy array.
        """
        if channel_name not in self._df.columns:
            raise KeyError(f"Channel '{channel_name}' not found in waveform data")
        return self._df[channel_name].to_numpy(dtype=float)