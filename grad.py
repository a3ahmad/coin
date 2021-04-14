import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import torchvision

import pytorch_lightning as pl

class COIN(pl.LightningModule):
    def __init__(self, num_hidden_layers, layer_width):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(2, layer_width),
        ] + [
            nn.Linear(layer_width, layer_width)
        ] * num_hidden_layers + [
            nn.Linear(layer_width, 3),
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.sin(x)

        return self.layers[-1](x)

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=2e-4)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.forward(x)
        loss = F.mse_loss(z, y)
        self.log('train_loss', loss)
        return loss

parser = argparse.ArgumentParser(description='COIN gradient descent implementation')
parser.add_argument("--hidden", type=int, default=13, help="Number of hidden layers")
parser.add_argument("--width", type=int, default=49, help="Hidden layer width")
parser.add_argument("--epochs", type=int, default=50000, help="Number of epochs")
parser.add_argument('--batch_size', type=int, help='Batch size')
parser.add_argument('--path', required=True, help='Path of image to compress')

args = parser.parse_args()

model = COIN(args.hidden, args.width)

image = torchvision.transforms.Compose([
    torchvision.transforms.ConvertImageDtype(torch.float),
    torchvision.transforms.Normalize(0, 255.0)
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
indices = torch.stack((norm_y, norm_x)).transpose(0, 1)
image_dataset = TensorDataset(indices, colors)
image_dataloader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=True)

trainer = pl.Trainer(
    gpus=1 if torch.cuda.is_available() else 0,
    max_epochs=args.epochs
    )
trainer.fit(model, image_dataloader)

image = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(0, 1.0 / 255.0),
    torchvision.transforms.ConvertImageDtype(torch.uint8),
])(model(indices).transpose(0,1).view(image.shape))
torchvision.io.write_png(image, args.path.replace('.png', '_compressed.png').replace('.jpg', '_compressed.png'))
