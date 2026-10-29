import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageDraw


class ViTLensImageDataset(Dataset):
    """
    Dataset wrapper that preserves image file paths, assigned index IDs,
    and ground-truth labels for downstream interpretability analysis.
    """
    def __init__(self, root_dir: str, transform=None, class_to_idx_mapping=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List of tuples: (image_path, local_class_idx, class_name)
        self.classes = []
        self.class_to_idx = {}
        self.imagenet1k_mapping = class_to_idx_mapping  # Optional mapping to 1000 ImageNet classes

        if os.path.exists(root_dir):
            self._scan_directory()

    def _scan_directory(self):
        """Scan directory formatted as root/class_name/xxx.ext"""
        entries = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        self.classes = entries
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(cls_dir, fname)
                    local_idx = self.class_to_idx[cls_name]
                    self.samples.append((full_path, local_idx, cls_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cls_name = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            tensor_image = transforms.ToTensor()(image)

        # ImageNet-1K target ID if mapped, else local dataset label
        target_1k = -1
        if self.imagenet1k_mapping and cls_name in self.imagenet1k_mapping:
            target_1k = self.imagenet1k_mapping[cls_name]

        return {
            "image": tensor_image,
            "label": label,
            "class_name": cls_name,
            "target_1k": target_1k,
            "path": path,
            "id": idx
        }


def get_vit_transforms(image_size: int = 224, is_train: bool = False):
    """
    Standard preprocessing matching ViT-B/16 ImageNet specifications.
    Resize -> CenterCrop -> ToTensor -> Normalize
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])


def build_dataloaders(data_dir: str,
                      batch_size: int = 32,
                      val_split: float = 0.2,
                      num_workers: int = 2,
                      image_size: int = 224,
                      class_mapping_path: str = None,
                      seed: int = 42):
    """
    Create train and validation DataLoaders with stratified or deterministic split.
    """
    mapping = None
    if class_mapping_path and os.path.exists(class_mapping_path):
        with open(class_mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

    full_dataset = ViTLensImageDataset(
        root_dir=data_dir,
        transform=get_vit_transforms(image_size, is_train=False),
        class_to_idx_mapping=mapping
    )

    if len(full_dataset) == 0:
        raise ValueError(f"No valid images found in directory: {data_dir}")

    total_len = len(full_dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, full_dataset


def create_synthetic_prototype_dataset(target_dir: str = "./data/prototype",
                                       num_classes: int = 5,
                                       images_per_class: int = 20,
                                       image_size: int = 224):
    """
    Generates a synthetic prototype dataset with geometric and texture patterns.
    Enables immediate offline validation and CPU smoke tests without multi-gigabyte downloads.
    """
    os.makedirs(target_dir, exist_ok=True)
    classes = [f"class_{i:03d}" for i in range(num_classes)]

    patterns = ["stripes", "circles", "grid", "checkerboard", "diagonal"]

    for c_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(target_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        pat_type = patterns[c_idx % len(patterns)]

        for img_idx in range(images_per_class):
            img = Image.new("RGB", (image_size, image_size), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)

            # Draw distinctive patterns
            if pat_type == "stripes":
                for y in range(0, image_size, 16):
                    draw.rectangle([0, y, image_size, y + 8], fill=(40 + c_idx * 30, 80, 180))
            elif pat_type == "circles":
                for r in range(20, 100, 18):
                    draw.ellipse([112 - r, 112 - r, 112 + r, 112 + r], outline=(200, 50, 50 + c_idx * 30), width=4)
            elif pat_type == "grid":
                for x in range(0, image_size, 20):
                    draw.line([x, 0, x, image_size], fill=(50, 180, 80), width=3)
                for y in range(0, image_size, 20):
                    draw.line([0, y, image_size, y], fill=(50, 180, 80), width=3)
            elif pat_type == "checkerboard":
                step = 28
                for x in range(0, image_size, step):
                    for y in range(0, image_size, step):
                        if (x // step + y // step) % 2 == 0:
                            draw.rectangle([x, y, x + step, y + step], fill=(30, 30, 30))
            else:  # diagonal
                for offset in range(-image_size, image_size, 24):
                    draw.line([offset, 0, offset + image_size, image_size], fill=(220, 150, 30), width=5)

            img.save(os.path.join(cls_dir, f"img_{img_idx:04d}.png"))

    return target_dir
