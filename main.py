import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.attn_simclr import Attn_SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook

from cluster_evaluation import Confusion


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j, mask = model(x_i, x_j, args.attn_head, args.mask)

        loss = criterion(z_i, z_j).to(args.device)
        loss.backward()

        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            if args.attn_head:
                writer.add_histogram("mask", mask, args.global_step)
            args.global_step += 1

        confusion, confusion_model = Confusion(10), Confusion(10)
        all_pred = all_prob.max(1)[1]
        confusion_model.add(all_pred, all_labels)
        confusion_model.optimal_assignment(args.num_classes)
        acc_model = confusion_model.acc()
        kmeans = cluster.KMeans(n_clusters=num_classes, random_state=args.seed)
        embeddings = all_embeddings.cpu().numpy()
        kmeans.fit(embeddings)
        pred_labels = torch.tensor(kmeans.labels_.astype(np.int))
        # clustering accuracy
        confusion.add(pred_labels, all_labels)
        confusion.optimal_assignment(args.num_classes)
        acc = confusion.acc()
        representation_cluster_score = confusion.clusterscores()
        model_cluster_score = confusion_model.clusterscores()

        loss_epoch += loss.item()
    return loss_epoch


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size, crop_size=args.crop_size),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size, crop_size=args.crop_size),
        )
    else:
        raise NotImplementedError

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    # initialize ResNet
    if args.model == "simclr":
        encoder = get_resnet(args.resnet, pretrained=False)
        n_features = encoder.fc.in_features  # get dimensions of fc layer
    else:
        encoder1 = get_resnet(args.resnet, pretrained=False)
        encoder2 = get_resnet(args.resnet, pretrained=False)
        n_features = encoder1.fc.in_features  # get dimensions of fc layer

    # initialize model
    if args.model == "simclr":
        model = SimCLR(encoder, args.projection_dim, n_features)
    else:
        model = Attn_SimCLR(encoder1, encoder2, args.projection_dim, n_features)

    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    str_lr = str(args.lr).replace('.', '')
    if args.nr == 0:
        if args.attn_head:
            if args.model == "attn_simclr":
                writer = SummaryWriter(os.path.join('runs', f"attn_simclr_{args.mask}_{args.dataset}_{args.epochs}_{args.resnet}_lr{str_lr}"))
            else:
                writer = SummaryWriter(os.path.join('runs', f"{args.mask}_{args.dataset}_{args.epochs}_{args.resnet}_lr{str_lr}"))
        else:
            writer = SummaryWriter(os.path.join('runs', f"simclr_{args.dataset}_{args.epochs}_{args.resnet}_lr{str_lr}"))

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    print(args)
    str_lr = str(args.lr).replace('.', '')
    print(f'name: {args.model}_{args.mask}_{args.dataset}_{args.epochs}_{args.resnet}_lr{str_lr}')


    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
