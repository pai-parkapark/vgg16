from base.base_trainer import BaseTrainer
from dataprovider.data_loader import CIFAR10DataLoader
from dataprovider.data_setter import CIFAR10DataSetter
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Subset
from model.vgg16.model import VGG16
from model.vgg16.loss import vgg16_loss
from model.vgg16.metric import vgg16_accuracy
import torch.optim as optim


class VGG16Trainer(BaseTrainer):

    def __init__(self, model, loss, optimizer, metric, train_data_loader, valid_data_loader, mac_gpu=True, *args,
                 **kwargs):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.mac_gpu = mac_gpu

        if self.mac_gpu:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.model = self.model.to(self.device)

    def _train_epoch(self, epoch):
        batch_loss = 0
        batch_total = 0
        batch_correct = 0
        self.model.train()

        for inputs, labels in tqdm(self.train_data_loader):
            if self.mac_gpu:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            main_outputs, aux_outputs = outputs
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            batch_loss += loss.item() * inputs.size(0)
            batch_total, batch_correct = self.metric(main_outputs, labels)

        epoch_loss = batch_loss / len(self.train_data_loader.dataset)
        epoch_accuracy = 100 * batch_correct / batch_total

        return epoch_loss, epoch_accuracy

    def train(self, epochs):
        print(f'{epochs} 번의 학습 시작')
        for epoch in range(epochs):
            epoch_loss, epoch_accuracy = self._train_epoch(epoch)
            print(f'Epoch: {epoch} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy}')
        print(f'{epochs} 번의 학습 완료')
        return epoch_loss, epoch_accuracy

    def validate(self):
        val_loss = 0
        total = 0
        correct = 0

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.valid_data_loader:
                if self.mac_gpu:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.loss(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_total, val_correct = self.metric(outputs, labels)
                total += val_total
                correct += val_correct
        val_acc = 100 * correct / total
        return val_loss, val_acc


if __name__ == '__main__':
    epochs = 1
    batch_size = 64

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # VGG16 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data_setter = CIFAR10DataSetter(root='./data', train=True, download=True, transform=transform)
    train_subset_indices = list(range(64 * 3))
    sub_train_dataset = Subset(train_data_setter, train_subset_indices)

    valid_data_setter = CIFAR10DataSetter(root='./data', train=False, download=True, transform=transform)
    valid_subset_indices = list(range(64 * 1))
    sub_valid_dataset = Subset(valid_data_setter, valid_subset_indices)

    train_data_loader = CIFAR10DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = CIFAR10DataLoader(sub_valid_dataset, batch_size=batch_size, shuffle=True)

    num_classes = 10
    model = VGG16(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = vgg16_loss
    metric_fn = vgg16_accuracy

    vgg16_trainer = VGG16Trainer(model=model,
                                 loss=loss_fn,
                                 optimizer=optimizer,
                                 metric=metric_fn,
                                 train_data_loader=train_data_loader,
                                 valid_data_loader=valid_data_loader,
                                 mac_gpu=True)

    train_loss, train_acc = vgg16_trainer.train(epochs)
    val_loss, val_acc = vgg16_trainer.validate()