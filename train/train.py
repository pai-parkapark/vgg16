from dataprovider.data_setter import CIFAR10DataSetter
from dataprovider.data_loader import CIFAR10DataLoader
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torch.optim as optim
from trainer import VGG16Trainer
from model.vgg16.model import VGG16
from model.vgg16.loss import vgg16_loss
from model.vgg16.metric import vgg16_accuracy


def train(model, loss_fn, metric_fn, epochs=20, batch_size=64, num_classes=10, lr=0.001):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # VGG16 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data_setter = CIFAR10DataSetter(root='./data', train=True, download=False, transform=transform)
    train_subset_indices = list(range(batch_size * 500))
    sub_train_dataset = Subset(train_data_setter, train_subset_indices)
    train_data_loader = CIFAR10DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=True)

    valid_data_setter = CIFAR10DataSetter(root='./data', train=False, download=False, transform=transform)
    valid_subset_indices = list(range(batch_size * 20))
    sub_valid_dataset = Subset(valid_data_setter, valid_subset_indices)
    valid_data_loader = CIFAR10DataLoader(sub_valid_dataset, batch_size=batch_size, shuffle=True)

    if model == 'vgg16':
        model = VGG16(num_classes=num_classes, aux_logits=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if loss_fn == 'vgg16':
        loss_fn = vgg16_loss

    if metric_fn == 'vgg16':
        metric_fn = vgg16_accuracy

    vgg16_v3_trainer = VGG16Trainer(model=model, loss=loss_fn, optimizer=optimizer, metric=metric_fn,
                                    train_data_loader=train_data_loader, valid_data_loader=valid_data_loader,
                                    mac_gpu=True)
    train_loss, train_acc = vgg16_v3_trainer.train(epochs)
    print(f"Trian Loss : {train_loss}, Trian accuracy : {train_acc}")

    val_loss, val_acc = vgg16_v3_trainer.validate()
    print(f"Validation Loss : {val_loss}, Validation accuracy : {val_acc}")


if __name__ == '__main__':
    train(model="vgg16",
          loss_fn="vgg16",
          metric_fn="vgg16",
          epochs=10,
          batch_size=64,
          num_classes=10,
          lr=0.001)
