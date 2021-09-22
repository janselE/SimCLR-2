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

# Concat dataset
from concat_dataset import ConcatDataset

# from scipy.optimize import linear_sum_assignment as hungarian
# from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)

import pandas as pd


def get_datasets(args, data):
    if data == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(
                size=args.image_size, crop_size=args.crop_size, attn_head=args.attn_head
            ),
        )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(
                size=args.image_size,
                crop_size=args.crop_size,
                is_training=False,
                attn_head=args.attn_head,
            ),
        )
    elif data == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(
                size=args.image_size, crop_size=args.crop_size, attn_head=args.attn_head
            ),
        )

        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(
                size=args.image_size,
                crop_size=args.crop_size,
                is_training=False,
                attn_head=args.attn_head,
            ),
        )
    else:
        raise NotImplementedError
    return train_dataset, test_dataset


def gen_embeddings(args, test_loader, model):
    str_lr = str(args.lr).replace(".", "")
    if args.attn_head:
        if args.model == "attn_simclr":
            path = f"attn_simclr_{args.mask}_{args.dataset}_{args.epochs}_{args.resnet}_lr{str_lr}"
        else:
            path = f"{args.mask}_{args.dataset}_{args.epochs}_{args.resnet}_lr{str_lr}"
    else:
        path = f"simclr_{args.dataset}_{args.epochs}_{args.resnet}_lr{str_lr}"

    x = None
    with torch.no_grad():
        for step, ((x_i), l) in enumerate(test_loader):
            if step % 10 == 0:
                print(step)
            x_i = x_i.cuda(non_blocking=True)
            h_i = model(x_i, None)
            x = h_i.cpu().numpy()
            y = l.cpu().numpy()
            y = np.expand_dims(y, axis=1)
            result = np.concatenate([y, x], 1)
            x_df = pd.DataFrame(result)
            x_df.to_csv(f"embeddings/{path}.csv", index=False, header=False, mode="a")


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    emb_nmi = []
    emb_ari = []
    emb_ami = []

    for step, (elements) in enumerate(train_loader):
        for data_name, e in zip(args.dataset, elements):
            dinput, labels = e
            x_i, x_j = dinput

            optimizer.zero_grad()
            x_i = x_i.to("cuda")
            x_j = x_j.to("cuda")

            h_i, h_j, z_i, z_j, mask = model(x_i, x_j, args.attn_head, args.mask)
            loss = criterion(z_i, z_j).to(args.device)

            loss.backward()

            optimizer.step()

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))

            # calculate the metrics
            embeddings_i = KMeans(n_clusters=10).fit(h_i.detach().cpu())
            embeddings_j = KMeans(n_clusters=10).fit(h_j.detach().cpu())

            pred_labels_i = embeddings_i.labels_
            pred_labels_j = embeddings_j.labels_

            # produce a numpy version of the labels
            all_labels = labels.detach().numpy()

            nmi_i = normalized_mutual_info_score(all_labels, pred_labels_i)
            nmi_j = normalized_mutual_info_score(all_labels, pred_labels_j)
            nmi = (nmi_i + nmi_j) / 2
            emb_nmi.append(nmi)

            ari_i = adjusted_rand_score(all_labels, pred_labels_i)
            ari_j = adjusted_rand_score(all_labels, pred_labels_j)
            ari = (ari_i + ari_j) / 2
            emb_ari.append(ari)

            ami_i = adjusted_mutual_info_score(all_labels, pred_labels_i)
            ami_j = adjusted_mutual_info_score(all_labels, pred_labels_j)
            ami = (ami_i + ami_j) / 2
            emb_ami.append(ami)

            writer.add_scalar(f"NMI/emb_train_epoch_{data_name}", nmi, args.global_step)
            writer.add_scalar(f"ARI/emb_train_epoch_{data_name}", ari, args.global_step)
            writer.add_scalar(f"AMI/emb_train_epoch_{data_name}", ami, args.global_step)

            if args.nr == 0 and step % 50 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

            if args.nr == 0:
                writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
                if args.attn_head:
                    writer.add_histogram("mask", mask, args.global_step)
                args.global_step += 1

            loss_epoch += loss.item()
    metrics = {"emb_ami": emb_ami, "emb_ari": emb_ari, "emb_nmi": emb_nmi}
    return loss_epoch, metrics


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.model == "simclr" and args.attn_head is False:
        args.crop_size = args.image_size
    print(f"image_size are {args.image_size}, {args.crop_size}")

    all_datasets_train = []
    all_datasets_test = []
    for data in args.dataset:
        train_dataset, test_dataset = get_datasets(args, data)
        all_datasets_train.append(train_dataset)
        all_datasets_test.append(test_dataset)

    if args.nodes > 1 and len(args.dataset) == 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            all_datasets_train[0], num_replicas=args.world_size, rank=rank, shuffle=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            all_datasets_test[0], num_replicas=args.world_size, rank=rank, shuffle=True
        )
    elif args.nodes > 1 and len(args.dataset) > 1:
        concat_train = ConcatDataset(tuple(all_datasets_train))
        concat_test = ConcatDataset(tuple(all_datasets_test))

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            concat_train, num_replicas=args.world_size, rank=rank, shuffle=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            concat_train, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None
        test_sampler = None

    concat_train = ConcatDataset(tuple(all_datasets_train))
    concat_test = ConcatDataset(tuple(all_datasets_test))

    train_loader = torch.utils.data.DataLoader(
        concat_train,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        concat_test,
        batch_size=args.batch_size,
        shuffle=(test_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=test_sampler,
    )

    # initialize ResNet
    if args.model == "simclr":
        encoder = get_resnet(args.resnet, pretrained=False)
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        model = SimCLR(encoder, args.projection_dim, n_features)
    else:
        encoder1 = get_resnet(args.resnet, pretrained=False)
        encoder2 = get_resnet(args.resnet, pretrained=False)
        n_features = encoder1.fc.in_features  # get dimensions of fc layer
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
    str_lr = str(args.lr).replace(".", "")
    if args.nr == 0:
        if args.attn_head:
            if args.model == "attn_simclr":
                writer = SummaryWriter(
                    os.path.join(
                        "runs",
                        f"attn_simclr_{args.mask}_{args.dataset}_{args.epochs}_{args.resnet}_lr{str_lr}",
                    )
                )
            else:
                writer = SummaryWriter(
                    os.path.join(
                        "runs",
                        f"{args.mask}_{args.dataset}_{args.epochs}_{args.resnet}_lr{str_lr}",
                    )
                )
        else:
            writer = SummaryWriter(
                os.path.join(
                    "runs",
                    f"simclr_{args.dataset}_{args.epochs}_{args.resnet}_lr{str_lr}",
                )
            )

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]["lr"]
        loss_epoch, metrics = train(
            args, train_loader, model, criterion, optimizer, writer
        )

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            writer.add_scalar(
                "NMI/emb_train",
                sum(metrics["emb_nmi"]) / len(metrics["emb_nmi"]),
                epoch,
            )
            writer.add_scalar(
                "ARI/emb_train",
                sum(metrics["emb_ari"]) / len(metrics["emb_ari"]),
                epoch,
            )
            writer.add_scalar(
                "AMI/emb_train",
                sum(metrics["emb_ami"]) / len(metrics["emb_ami"]),
                epoch,
            )

            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)
    gen_embeddings(args, test_loader, model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    str_lr = str(args.lr).replace(".", "")
    print(
        f"name: {args.model}_{args.mask}_{args.dataset}_{args.epochs}_{args.resnet}_lr{str_lr}"
    )

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    print("Device is: ", args.device)

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)

