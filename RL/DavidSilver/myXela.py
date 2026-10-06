import numpy as np
import json
import os
import time
from typing import Dict, List
from pathlib import Path
import csv

class XelaTactileFeatureExtractor:
   # Class-level constants
    # class-level cache
    _MEAN_REST = None
    _STD_REST = None
    _NORM_FAST_XELA_ABS_MAX = None
    # dataset path relative to THIS file
    _DATASET_PATH = Path(__file__).resolve().parent / "xelaDataset"

    def __init__(self, force_threshold=50):
        self.force_threshold = force_threshold
        self.special = None

        # Load constants once
        self._load_rest_stats()

        self.mean_rest = self._MEAN_REST
        self.std_rest = self._STD_REST

        # print(
        #     f"Initialized XelaTactileFeatureExtractor "
        #     f"with mean_rest shape: {self.mean_rest.shape}, "
        #     f"std_rest shape: {self.std_rest.shape}"
        # )

    @classmethod
    def _load_rest_stats(cls):
        """Load mean/std only once (lazy + safe)."""
        if cls._MEAN_REST is not None:
            return

        mean_path = cls._DATASET_PATH / "mean_rest.npy"
        std_path = cls._DATASET_PATH / "sigma_rest.npy"
        norm_fast_xela_abs_max_path = cls._DATASET_PATH / "norm_fast_xela_abs_max.npy"

        if not mean_path.exists():
            raise FileNotFoundError(f"Missing file: {mean_path}")
        if not std_path.exists():
            raise FileNotFoundError(f"Missing file: {std_path}")

        cls._MEAN_REST = np.load(mean_path)
        cls._STD_REST = np.load(std_path)
        cls._NORM_FAST_XELA_ABS_MAX = np.array([971.8997973376092, 4670.376938531569, 1717.4039052922467])  #np.load(norm_fast_xela_abs_max_path)

    def extract_force(self, frame_dict):
        sensor = frame_dict['1']
        special = np.array(sensor['special'])

        # Core signals
        
        Fx = special[:, 0]
        Fy = special[:, 1]
        Fz = special[:, 2]
        temperature = special[:, 3]
        normal_force = special[:, 6]
        pressure = special[:, 11]
        self.fx_raw, self.fy_raw, self.fz_raw = Fx, Fy, Fz
        self.temperature = temperature
        self.norm_special= np.abs((special[:,:3] - self.mean_rest) / self.std_rest) # normalize --> value - mean / std
        self.norm_special = self.norm_special / self._NORM_FAST_XELA_ABS_MAX  # scale to max observed
        self.fx_norm, self.fy_norm, self.fz_norm = self.norm_special[:, 0], self.norm_special[:, 1], self.norm_special[:, 2]
        
        
    def normalize_force(self):
        
        pass




class XelaTactileRecorder:
    def __init__(
        self,
        save_every: int = 100,
        save_dir: str="xelaDataset",
        file_prefix: str = "xela_record",
        auto_timestamp: bool = True
    ):
        """
        Parameters
        ----------
        save_dir : str
            Directory where JSON files will be stored
        save_every : int
            Number of timesteps per saved file
        file_prefix : str
            Prefix for saved JSON files
        auto_timestamp : bool
            Append wall-time to filename if True
        """
        self.save_dir = save_dir
        self.save_every = save_every
        self.file_prefix = file_prefix
        self.auto_timestamp = auto_timestamp

        self.buffer: List[Dict] = []
        self.file_counter = 0

        os.makedirs(self.save_dir, exist_ok=True)

    def record(self, frame_dict: Dict):
        """
        Record a single timestep
        """
        self.buffer.append(frame_dict)

        if len(self.buffer) >= self.save_every:
            self._flush()

    def _flush(self):
        """
        Save buffered frames to disk
        """
        if not self.buffer:
            return

        timestamp = int(time.time())
        if self.auto_timestamp:
            filename = (
                f"{self.file_prefix}_"
                f"{self.file_counter:04d}_"
                f"{timestamp}.json"
            )
        else:
            filename = f"{self.file_prefix}_{self.file_counter:04d}.json"

        path = os.path.join(self.save_dir, filename)

        with open(path, "w") as f:
            json.dump(self.buffer, f, indent=2)

        self.buffer.clear()
        self.file_counter += 1

    def close(self):
        """
        Flush remaining data before shutdown
        """
        self._flush()


class XelaTactileCSVLogger:
    def __init__(self, csv_file: str = "dataset_Xela/xela_data.csv"):
        """
        Parameters
        ----------
        csv_file : str
            Path to the CSV file where data will be logged.
        """
        self.csv_file = csv_file

        # Ensure the directory path exists before creating the file
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Header includes timestamp and force_X, force_Y, force_Z
                header = ["timestamp"] + [f"force_X_{i}" for i in range(16)] + [f"force_Y_{i}" for i in range(16)] + [f"force_Z_{i}" for i in range(16)]
                writer.writerow(header)

    def log_forces(self, force_X: np.ndarray, force_Y: np.ndarray, force_Z: np.ndarray):
        """
        Log force_X, force_Y, and force_Z (each as a (16,) numpy array) along with the current system time to the CSV file.

        Parameters
        ----------
        force_X : np.ndarray
            A numpy array of shape (16,) representing the X components of the forces.
        force_Y : np.ndarray
            A numpy array of shape (16,) representing the Y components of the forces.
        force_Z : np.ndarray
            A numpy array of shape (16,) representing the Z components of the forces.
        """
        if force_X.shape != (16,) or force_Y.shape != (16,) or force_Z.shape != (16,):
            raise ValueError("All force arrays must have a shape of (16,).")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp] + force_X.tolist() + force_Y.tolist() + force_Z.tolist())