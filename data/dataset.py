"""Dataset helpers for NRRD volumes."""
from monai.data import Dataset

class NRRDDataset(Dataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["src_path"] = sample.get("image_meta_dict", {}).get("filename_or_obj")
        return sample
