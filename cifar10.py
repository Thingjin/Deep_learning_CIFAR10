from gc import callbacks
import torch
from torch import nn
# from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR


# USE_CUDA = torch.cuda.is_available()
# print(USE_CUDA)

# device = torch.device('cuda:0' if USE_CUDA else 'cpu')
# print('학습을 진행하는 기기:',device)

# print('cuda index:', torch.cuda.current_device())

# print('gpu 개수:', torch.cuda.device_count())

# print('graphic name:', torch.cuda.get_device_name())

# cuda = torch.device('cuda')

# print(cuda)

#출력결과----------------------------------
#True
#학습을 진행하는 기기: cuda:0
#cuda index: 0
#gpu 개수: 2
#graphic name: NVIDIA RTX A5000
#cuda
# ==> gpu 실행 확인
#------------------------------------------

#torch.randn(5).cuda() #난수 설정

transformer = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    ]
)

# train_data = datasets.CIFAR10('/home/jovyan/Destop/data', train = True, download = True, transform=transformer)
# test_data = datasets.CIFAR10('/home/jovyan/Destop/data', train = False, download = True, transform=transformer)


#model
#ligthning module을 정의하기 위해 LightningModule 클래스를 상속받고 모델, training, validation, test 그리고 optimizer 등을 구현
class CIFARClassifier(pl.LightningModule):
    def __init__(self, lr = 0.05, num_classes = 10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.c2 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) # 16x16
        self.c3 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.c4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) # 8x8
               
        self.c_d1 = nn.Linear(in_features=8*8*64, out_features=256)
        self.c_d1_bn = nn.BatchNorm1d(256)
        self.c_d1_drop = nn.Dropout(0.5)
        
        self.c_d2 = nn.Linear(in_features=256, out_features=10)


    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.pool1(x) # 16
        
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = self.pool2(x) # 8
    
        batch_size = x.size(0)     
        x = F.relu(self.c_d1(x.view(batch_size, -1)))
        #x = self.c_d1_bn(x)
        #x = self.c_d1_drop(x)
        
        x = self.c_d2(x)

        x = F.log_softmax(x, dim=1)
        return x

    #optimizer
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr = 0.025, momentum = 0.9)
        #optimizer = torch.optim.Adam(self.parameters(), lr = 0.05)
        return optimizer

    #cross_entroy_loss
    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    #training_step
    def training_step(self, train_batch):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, device):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('evl_loss', loss)
        return loss

class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='/home/jovyan/Destop/data', batch_size = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transformer

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train= True, download = True)
        datasets.CIFAR10(self.data_dir, train=False, download = True)


    def setup(self, stage =None):
        if stage == 'fit' or stage is None:
            train_data = datasets.CIFAR10(self.data_dir, train=True, transform=transformer)
            self.train_data, self.val_data = random_split(train_data, [45000,5000])
        if stage == 'test' or stage is None:
            self.test_data = datasets.CIFAR10(self.data_dir, train=False, transform = transformer)

    def train_dataloader(self):
        train_data = DataLoader(self.train_data, batch_size = self.batch_size)
        return train_data
    
    def val_dataloader(self):
        val_data = DataLoader(self.val_data, batch_size = self.batch_size)
        return val_data

    def test_dataloader(self):
        test_data = DataLoader(self.test_data, batch_size =self.batch_size)
        return test_data

# #DataLoader
# train_loader = DataLoader(
#     dataset = train_data,
#     batch_size = 10,
#     shuffle = False, sampler=None, batch_sampler=None,
#     num_workers = 0, collate_fn=None,
#     pin_memory=False, drop_last=False, timeout=0,
#     worker_init_fn=None
# )


cifar10 = CIFARDataModule()

model = CIFARClassifier()

import wandb
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project = '2018125041_이명진_CIFAR10')


from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar

trainer = pl.Trainer(
    logger = wandb_logger,
    accelerator = "gpu",
    devices = 2,
    max_epochs = 20,
    strategy = 'dp',
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],)

trainer.fit(model, datamodule=cifar10)
trainer.test(model, datamodule=cifar10)
