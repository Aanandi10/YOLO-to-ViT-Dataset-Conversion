# YOLO → ViT dataset conversion

## Install
```bash
pip install pyyaml pandas pillow            
pip install torch torchvision transformers  
```

## Run
```bash
python convert_yolo_to_vit.py --input /path/to/dataset --output /path/to/vit_dataset
# options: --symlink  --overwrite  --empty-as-class NO_KRISHNA  --data-yaml PATH
python verify_vit_dataset.py --dataset /path/to/vit_dataset
```

## Rules applied (in order, per image)
| Condition | Result |
|---|---|
| no `labels/<split>/<stem>.txt` | excluded: `missing_annotation_file` |
| malformed line or class ID not in `data.yaml` | excluded: `invalid_annotation` |
| empty label file | excluded: `empty_annotation_file` (only mapped to a class if you pass `--empty-as-class`) |
| one distinct class ID (any number of boxes) | included |
| 2+ distinct class IDs | excluded: `ambiguous_multiple_classes` |
| PIL cannot open image | excluded: `corrupted_image` |

Every exclusion is listed in `metadata/excluded_images.csv`. Splits come only from the source directory. Images are never moved between splits.

## Output
```
vit_dataset/
├── train|val|test/<ClassName>/image files     # ImageFolder-ready, all class folders in every split
├── metadata/
│   ├── {train,val,test}_labels.csv            # filename, split, class_id, class_name, ...
│   ├── class_mapping.json                     # YOLO id <-> folder <-> ImageFolder index
│   ├── conversion_summary.csv / conversion_report.txt
│   ├── ambiguous_images.csv, missing_annotations.csv (missing + empty), invalid_annotations.csv
│   ├── excluded_images.csv                    # every excluded image + reason
│   ├── orphan_label_files.csv                 # label files with no image
│   └── cross_split_duplicates.csv             # identical bytes across splits (leakage warning)
└── dataset_config.json
```

**Index note:** `ImageFolder` numbers classes alphabetically by folder name, so its indices can differ from the YOLO class IDs. Both are stored in `class_mapping.json` and the CSVs. Pick one convention and use it consistently.

## Load with PyTorch
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

tf_train = transforms.Compose([
    transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])   # ViT-B/16 (google/vit-base-patch16-224) uses 0.5/0.5
tf_eval = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

root = "vit_dataset"
# allow_empty=True is needed if a class has zero images in some split (torchvision >= 0.15)
train_ds = datasets.ImageFolder(f"{root}/train", tf_train, allow_empty=True)
val_ds   = datasets.ImageFolder(f"{root}/val",   tf_eval,  allow_empty=True)
test_ds  = datasets.ImageFolder(f"{root}/test",  tf_eval,  allow_empty=True)
assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=4)

# Hugging Face head labels
id2label = {v: k for k, v in train_ds.class_to_idx.items()}
label2id = train_ds.class_to_idx
```
Prefer the checkpoint's own `AutoImageProcessor` mean/std if you use a different ViT.
