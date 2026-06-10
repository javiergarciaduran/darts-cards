# datasets/cards.py
import torchvision.datasets as dset
import torchvision.transforms as T


CARDS_MEAN = [0.7790, 0.7314, 0.7053]
CARDS_STD  = [0.2703,  0.2972,  0.3059]


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


def get_cards(data_path, input_size=32, cutout_length=0, split='train'):
    training = (split == 'train')
    dataset = dset.ImageFolder(
        root=f"{data_path}/{split}",
        transform=build_transforms(input_size, training=training, cutout_length=cutout_length)
    )
    n_classes = len(dataset.classes)
    return dataset, n_classes
