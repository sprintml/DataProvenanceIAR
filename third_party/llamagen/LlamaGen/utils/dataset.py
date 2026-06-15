from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import numpy as np

class CocoGroundTruthDataset(Dataset):
    def __init__(self, image_dir, image_size=256, num_samples=40000):
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not self.image_files:
             raise ValueError(f"No images found in directory {image_dir}")
        print(f"Found {len(self.image_files)} images in {image_dir}")
        #random.shuffle(self.image_files) if num_samples > 0 else self.image_files
        self.image_files= self.image_files[:num_samples] if num_samples > 0 else self.image_files
    def __len__(self):
        return len(self.image_files)

    def center_crop_arr(self, pil_image, image_size):
        width, height = pil_image.size
        crop_size = min(width, height)
        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2
        pil_image = pil_image.crop((left, top, right, bottom))
        pil_image = pil_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        return pil_image

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            pil_image = Image.open(img_path).convert("RGB")
            pil_image = self.center_crop_arr(pil_image, self.image_size)
            image = load_image_to_tensor(pil_image)
            return image.unsqueeze(0), None
        except Exception as e:
            print(f"Error processing image {img_path}: {e}. Returning a placeholder.")
            return self.__getitem__((idx -1))

def load_image_to_tensor(image_path, target_device='cpu'):
    try:
        pil_image = Image.open(image_path).convert('RGB')
        img_np_uint8 = np.array(pil_image)
        img_np_float01 = img_np_uint8.astype(np.float32) / 255.0
        img_np_float_neg1_pos1 = 2.0 * img_np_float01 - 1.0
        tensor_hwc = torch.from_numpy(img_np_float_neg1_pos1)
        tensor_chw = tensor_hwc.permute(2, 0, 1)
        reconstructed_tensor = tensor_chw.to(target_device).float()
        return reconstructed_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None