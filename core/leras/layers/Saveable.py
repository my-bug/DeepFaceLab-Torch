import pickle
from pathlib import Path
from core import pathex
import numpy as np
import torch

from core.leras.nn import nn

class Saveable():
    def __init__(self, name=None):
        self.name = name

    #override
    def get_weights(self):
        #return torch parameters that should be initialized/loaded/saved
        return []

    #override
    def get_weights_np(self):
        weights = self.get_weights()
        if len(weights) == 0:
            return []
        return [w.detach().cpu().numpy() for w in weights]

    def set_weights(self, new_weights):
        weights = self.get_weights()
        if len(weights) != len(new_weights):
            raise ValueError ('len of lists mismatch')

        for w, new_w in zip(weights, new_weights):
            if isinstance(new_w, torch.nn.Parameter) or isinstance(new_w, torch.Tensor):
                src = new_w.data if hasattr(new_w, 'data') else new_w
                src = src.to(device=w.device, dtype=w.dtype)
                w.data.copy_(src)
            else:
                if not isinstance(new_w, np.ndarray):
                    new_w = np.array(new_w)
                src = torch.from_numpy(new_w).reshape(w.shape).to(device=w.device, dtype=w.dtype)
                w.data.copy_(src)

    def save_weights(self, filename, force_dtype=None):
        d = {}
        weights = self.get_weights()

        if self.name is None:
            raise Exception("name must be defined.")

        name = self.name

        for i, w in enumerate(weights):
            w_val = w.detach().cpu().numpy().copy()
            
            if force_dtype is not None:
                w_val = w_val.astype(force_dtype)

            w_name = f"param_{i}"
            d[w_name] = w_val

        # Stream pickle directly to disk to avoid an extra in-memory copy
        # of the entire weights dict (pickle.dumps), which can trigger
        # MemoryError on large models.
        p = Path(filename)
        p_tmp = p.parent / (p.name + '.tmp')
        with open(p_tmp, 'wb') as f:
            pickle.dump(d, f, protocol=4)
        if p.exists():
            p.unlink()
        p_tmp.rename(p)

    def load_weights(self, filename):
        """
        returns True if file exists
        """
        filepath = Path(filename)

        if not filepath.exists():
            # Compatibility: older DFL models often used .npy filenames,
            # while this repo may use .pth filenames (or vice versa).
            # The underlying format here is pickle of numpy arrays, so
            # swapping extensions is safe and preserves behavior.
            alt = None
            if filepath.suffix == '.pth':
                alt = filepath.with_suffix('.npy')
            elif filepath.suffix == '.npy':
                alt = filepath.with_suffix('.pth')
            if alt is not None and alt.exists():
                filepath = alt

        if filepath.exists():
            # Stream unpickle to avoid reading the whole file into memory first.
            with open(filepath, 'rb') as f:
                d = pickle.load(f)
        else:
            return False

        weights = self.get_weights()

        if self.name is None:
            raise Exception("name must be defined.")

        try:
            for i, w in enumerate(weights):
                w_name = f"param_{i}"
                w_val = d.get(w_name, None)

                if w_val is None:
                    # Weight not found, keep current initialization
                    pass
                else:
                    w_val = np.reshape(w_val, w.shape)
                    src = torch.from_numpy(w_val).to(device=w.device, dtype=w.dtype)
                    w.data.copy_(src)
        except:
            return False

        return True

    def init_weights(self):
        # PyTorch initializes weights automatically
        pass

nn.Saveable = Saveable
