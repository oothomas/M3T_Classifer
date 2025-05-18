"""Dataset helpers for NRRD volumes."""

from monai.data import Dataset


class NRRDDataset(Dataset):
    """Dataset that records the source path for each sample.

    This class simply extends :class:`monai.data.Dataset` to keep track of the
    file path from which each sample was loaded. The path is stored under the
    key ``"src_path"`` so it can be accessed during training or inference.
    """

    def __getitem__(self, index):
        """Return a single sample with an additional ``src_path`` field."""
        sample = super().__getitem__(index)
        sample["src_path"] = sample.get("image_meta_dict", {}).get("filename_or_obj")
        return sample
