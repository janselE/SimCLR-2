import os
import torch

from simclr import SimCLR
from simclr.modules import LARS


def load_optimizer(args, model):

    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#lr=3e-4)  # TODO: LARS
    elif args.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def save_model(args, model, optimizer):

    if args.attn_head:
        if args.model == "attn_simclr":
            path = os.path.join(args.model_path, f"attn_simclr_{args.mask}_{args.dataset}_{args.epochs}_{args.resnet}")
        else:
            path = os.path.join(args.model_path, f"{args.mask}_{args.dataset}_{args.epochs}_{args.resnet}")
    else:
        path = os.path.join(args.model_path, f"simclr_{args.dataset}_{args.epochs}_{args.resnet}")
        print(path)

    if not os.path.exists(path):
        os.makedirs(path)

    out = os.path.join(path, f"checkpoint_{args.current_epoch}.tar")


    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)
