import numpy as np
import h5py

class NYUDepthDataset:
    def __init__(self, path):
        self.path = path

        with h5py.File(self.path, 'r') as f:
            self.images = np.array(f['images']).transpose(0, 1, 3, 2) 
            self.depths = np.array(f['depths']).transpose(0, 2, 1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.depths[idx]