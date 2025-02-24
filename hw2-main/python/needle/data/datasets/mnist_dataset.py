from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        import gzip
        with gzip.open(image_filename, 'rb') as f:
            data = f.read()
            num_images = int.from_bytes(data[4:8], byteorder='big')
            num_rows = int.from_bytes(data[8:12], byteorder='big')
            num_cols = int.from_bytes(data[12:16], byteorder='big')
            self.num_rows = num_rows
            self.num_cols = num_cols
            images = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_images, num_rows * num_cols)
            images = images.astype(np.float32) / 255.0

        with gzip.open(label_filename, 'rb') as f:
            data = f.read()
            labels = np.frombuffer(data[8:], dtype=np.uint8)

        self.images = images
        self.labels = labels
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image = self.images[index]
        label = self.labels[index]
        if self.transforms is not None:
            image = np.reshape(image, (self.num_rows, self.num_cols, -1))
            image = self.apply_transforms(image)
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION