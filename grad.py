import argparse
import datetime
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import torchvision

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# from pytorch_msssim import ssim

# from sam import sam


OMEGA = 30.0

parser = argparse.ArgumentParser(description='COIN gradient descent implementation')
parser.add_argument("--hidden", type=int, default=13, help="Number of hidden layers")
parser.add_argument("--width", type=int, default=49, help="Hidden layer width")
parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
parser.add_argument("--epochs", type=int, default=50000, help="Number of epochs")
parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle each batch")
parser.add_argument('--batch_size', type=int, help='Batch size')
parser.add_argument('--path', required=True, help='Path of image to compress')

args = parser.parse_args()

image = torchvision.transforms.Compose([
    torchvision.transforms.ConvertImageDtype(torch.float),
])(torchvision.io.read_image(args.path))

# Default batch_size is the entire image
if args.batch_size is None:
    args.batch_size = image.shape[-1] * image.shape[-2]

half_width = image.shape[-1] // 2
half_height = image.shape[-2] // 2
indices = torch.where(image[0, ...] >= 0)
colors = image[:, indices[0], indices[1]].transpose(0, 1)
norm_y = (indices[0].float() - half_height) / half_height
norm_x = (indices[1].float() - half_width) / half_width
indices = torch.stack((norm_y, norm_x)).transpose(0, 1).cuda()
image_dataset = TensorDataset(torch.zeros(size=(1,1)), torch.zeros(size=(1,1)))
image_dataloader = DataLoader(image_dataset, batch_size=1)

def test_compress(name = None):
    global indices
    global image

    regen_image = torchvision.transforms.Compose([
        torchvision.transforms.ConvertImageDtype(torch.uint8),
    ])(model(indices).transpose(0,1).view(image.shape))
    if name is None:
        torchvision.io.write_png(regen_image.cpu(), args.path.replace('.png', '_compressed.png').replace('.jpg', '_compressed.png'))
    else:
        torchvision.io.write_png(regen_image.cpu(), args.path.replace('.png', '_compressed.png').replace('.jpg', '_compressed.png').replace('.', f'_{name}.'))


class COIN(pl.LightningModule):
    def __init__(self, num_hidden_layers, layer_width, lr, shuffle, indices, colors):
        super().__init__()

        self.register_buffer('x', indices)
        self.register_buffer('y', colors)
        self.lr = lr
        self.shuffle = shuffle

        self.layers = nn.ModuleList(self.init_weights([
            nn.Linear(2, layer_width),
        ], 1.0 / 2.0) + self.init_weights([
            nn.Linear(layer_width, layer_width),
        ] * num_hidden_layers, math.sqrt(6.0 / layer_width) / OMEGA) + self.init_weights([
            nn.Linear(layer_width, 3),
        ], math.sqrt(6.0 / layer_width) / OMEGA))

    def init_weights(self, linear_layers, scale):
        for layer in linear_layers:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -scale, scale)

        return linear_layers

    def forward(self, x):
        x = self.layers[0](x)
        x = torch.sin(OMEGA * x)

        for layer in self.layers[1:-1]:
            x = layer(x)
            x = torch.sin(OMEGA * x)

        x = self.layers[-1](x)
        x = torch.sigmoid(x)

        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'train_loss'
        }
        return [optimizer], [scheduler]

        #optimizer = sam.SAM(self.parameters(), torch.optim.SGD, lr=self.lr, momentum=0.9)
        #return optimizer

    def training_epoch_end(self, outputs):
        if self.trainer.current_epoch % 1000 == 0:
            test_compress(self.trainer.current_epoch)

    def training_step(self, train_batch, batch_idx):
        if self.shuffle:
            idx = torch.randperm(self.x.shape[0])
            z = self.forward(self.x[idx])
            loss = F.mse_loss(z, self.y[idx])

            self.log('train_loss', loss)
            return loss
        else:
            z = self.forward(self.x)
            loss = F.mse_loss(z, self.y)

            self.log('train_loss', loss)
            return loss


model = COIN(args.hidden, args.width, args.lr, args.shuffle, indices, colors)

trainer = pl.Trainer(
    gpus=1 if torch.cuda.is_available() else 0,
    logger=WandbLogger(project="COIN", id=f"{OMEGA}-{args.hidden}-{args.width}-{args.lr}-{datetime.datetime.now().strftime('%d-%m-%y-%Ih%Mmin')}"),
#    precision=16 if torch.cuda.is_available() else 32,
    max_epochs=args.epochs
    )

trainer.fit(model, image_dataloader)

indices = indices.cpu()
test_compress()