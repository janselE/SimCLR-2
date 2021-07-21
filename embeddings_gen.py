import os
import numpy as np
import torch
import torchvision


from simclr import SimCLR
from simclr.attn_simclr import Attn_SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from model import load_optimizer, save_model


import pandas as pd


train_dataset = torchvision.datasets.CIFAR10(
    "./datasets",
    download=True,
    transform=TransformsSimCLR(size=224, crop_size=224, is_training=False),
)

train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=(train_sampler is None),
    drop_last=True,
    num_workers=1,
    sampler=train_sampler,
)

# initialize ResNet
#encoder = get_resnet("resnet50", pretrained=False)
#n_features = encoder.fc.in_features  # get dimensions of fc layer

encoder1 = get_resnet("resnet50", pretrained=False)
encoder2 = get_resnet("resnet50", pretrained=False)
n_features = encoder1.fc.in_features  # get dimensions of fc layer

# initialize model
#model = SimCLR(encoder, 64, n_features)
model = Attn_SimCLR(encoder1, encoder2, 64, n_features)

#    "save_models/simclr_CIFAR10_100_resnet50", "checkpoint_90.tar"
model_fp = os.path.join(
    "save_models/attn_simclr_hard_CIFAR10_50_resnet50", "checkpoint_50.tar"
)
model.load_state_dict(torch.load(model_fp))#map_location=args.device.type))

#model = model.to(args.device)

# optimizer / loss
#optimizer, scheduler = load_optimizer(args, model)
#criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)


#model = model.to(args.device)


x = None
for step, ((x_i), l) in enumerate(train_loader):
        h_i = model(x_i, None)
        x = h_i.detach().numpy()
        y = l.detach().numpy()
        y = np.expand_dims(y, axis=1)
        result = np.concatenate([y, x], 1)
        x_df = pd.DataFrame(result)
        x_df.to_csv("tmp.csv", index=False, header=False, mode='a')

