import os
import json
import gc
import numpy as np


def get_multiple_profiles_path(path, num_profiles, seed=999):
    """Get multiple profiles from the same directory.
        Given a path, sample randomly num_profiles times from the directory.
    """
    print(f">>>> Getting {num_profiles} profiles from {path}")
    files = os.listdir(path)
    files = [f for f in files if f.endswith(".csv")]
    files = [os.path.join(path, f) for f in files]
    files = sorted(files)
    np.random.seed(seed)
    files = np.random.choice(files, num_profiles, replace=False)
    return files