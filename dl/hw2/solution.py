# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys

# List any extra packages you need here. Please, fix versions so reproduction of your results would be less painful.
PACKAGES_TO_INSTALL = ["gdown==4.4.0",]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models
from torchvision.transforms import v2
import wandb
import cv2
import os


class ImageTransform:
    def __init__(self, kind):
        if kind == 'train':
            self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                      v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.GaussianNoise(sigma=0.01),
                                      v2.RandomRotation(15), 
                                      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
        else:
            self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def __call__(self, sample):
        return self.transform(sample)


class CustomImageDataset(Dataset):
    def __init__(self, path, kind, transform=None, target_transform=None):
        self.prepare_image_path(path, kind)
        self.transform = transform
        self.target_transform = target_transform

    def prepare_image_path(self, path, kind):
        dataset_path = os.path.join(path, kind)
        if kind == "train":
            self.class_to_idx = {}
            image_path_list = []
            label_list = []
            class_list = os.listdir(dataset_path)
            class_list.sort()
            for i, cls in enumerate(class_list):
                self.class_to_idx[cls] = i
                cls_images_path = os.path.join(dataset_path, cls + '/images')
                image_list = os.listdir(cls_images_path)
                for image_name in image_list:
                    image_path_list.append(os.path.join(cls_images_path, image_name))
                label_list += [i] * len(image_list)
            self.classes = list(range(len(class_list)))
        elif kind == "val":
            image_path_list = []
            label_list = []
            class_list = os.listdir(dataset_path)
            class_list.sort()
            for i, cls in enumerate(class_list):# to work properly, needed to create map name to int, using train info data
                cls_images_path = os.path.join(dataset_path, cls)
                image_list = os.listdir(cls_images_path)
                for image_name in image_list:
                    image_path_list.append(os.path.join(cls_images_path, image_name))
                label_list += [i] * len(image_list)

        elif kind == "test":
            image_path_list = []
            label_list = []
            images_path = os.path.join(dataset_path, 'images')
            image_list = os.listdir(images_path)
            for image_name in image_list:
                image_path_list.append(os.path.join(images_path, image_name))
            label_list = [-1] * len(image_list)

        else:
            raise Exception("wrong kind argument!!!")
    
        self.imgs = list(zip(image_path_list, label_list))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = cv2.imread(self.imgs[idx][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.imgs[idx][1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val' or 'test', the dataloader should be deterministic.
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train', 'val' or 'test'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    dataset = CustomImageDataset(path, kind, ImageTransform(kind), torch.tensor)
    shuffle = True if kind == 'train' else False
    return DataLoader(dataset, batch_size=64, num_workers=4, shuffle=shuffle)

def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    model = models.resnet50(weights=None, num_classes=200)
    model.to('cuda')
    return model

def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    return optimizer

def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    return model(batch.to('cuda'))

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    accuracy = 0
    total_loss = 0
    n = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for X, y in dataloader:
            y = y.to('cuda')
            out = model(X.to('cuda'))
            loss = criterion(out, y)
            accuracy += (out.argmax(1) == y).sum().item()
            total_loss += loss.item()
            n += X.shape[0]
    
    return accuracy / n, loss / n


def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    EPOCH_NUM = 40
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    wandb.login(key="c86834f3fe7719b70c289274009689f98e6f5c1d")
    wandb.init(
        project="aim_ml3_hw2",

        config={
        "learning_rate": "0.001+scheduler",
        "architecture": "resnet50",
        "dataset": "tiny-imagenet-200",
        "epochs": EPOCH_NUM,
        "batch_size": 64,
        "optim": "Adam",
        "transform": "flip+noise"
        }
    )

    best_val_acc = 0
    for i in range(EPOCH_NUM):
        total_acc = 0
        total_loss = 0
        total_n = 0
        model.train()
        for X, y in train_dataloader:
            X = X.to('cuda')
            y = y.to('cuda')
            
            out = model(X)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = (out.argmax(1) == y).sum().item()
            total_acc += accuracy
            total_loss += loss.item() * X.shape[0]
            total_n += X.shape[0]

        train_acc = total_acc / total_n
        train_loss = total_loss / total_n

        scheduler.step()
        model.eval()
        val_acc, val_loss = validate(val_dataloader, model)
        
        if (val_acc > best_val_acc):
            best_val_acc = val_acc
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': best_val_acc,
            }, 'checkpoint.pth')
        
        wandb.log({"train_acc": train_acc, "train_loss": train_loss, 
               "val_acc": val_acc, "val_loss": val_loss})

    wandb.finish()
    best_weight = torch.load('checkpoint.pth', weights_only=True)
    model.load_state_dict(best_weight['model_state_dict'])


def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    weights = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(weights['model_state_dict'])
    model.to('cuda')

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """ 
    md5_checksum = "1089d094ad7bf904c14058d4ff7103b6"
    google_drive_link = "https://drive.google.com/file/d/1F10BDQ-lvf749rfDA7TJIFvMXTwZ3Al5/view?usp=sharing"

    return md5_checksum, google_drive_link
