# datasets/cards.py
import torchvision.datasets as dset
import torchvision.transforms as T

# Replace with your computed values from compute_stats.py
CARDS_MEAN = [0.485, 0.456, 0.406]
CARDS_STD  = [0.229, 0.224, 0.225]


def build_transforms(input_size=32, training=True, cutout_length=0):
    if training:
        tfms = [
            T.Resize(int(input_size * 1.15)),
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(CARDS_MEAN, CARDS_STD),
        ]
    else:
        tfms = [
            T.Resize(int(input_size * 1.15)),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(CARDS_MEAN, CARDS_STD),
        ]
    if training and cutout_length > 0:
        from preproc import Cutout
        tfms.append(Cutout(cutout_length))
    return T.Compose(tfms)


def get_cards(data_path, input_size=32, cutout_length=0):
    train_data = dset.ImageFolder(
        root=f"{data_path}/train",
        transform=build_transforms(input_size, training=True, cutout_length=cutout_length)
    )
    val_data = dset.ImageFolder(
        root=f"{data_path}/val",
        transform=build_transforms(input_size, training=False)
    )
    n_classes = len(train_data.classes)
    return train_data, val_data, n_classes
