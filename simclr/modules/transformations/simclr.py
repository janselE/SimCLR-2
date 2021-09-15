import torchvision


class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, crop_size, is_training=True, attn_head=False):
        self.is_training = is_training
        self.attn_head = attn_head
        print("attn_head")
        print(self.attn_head)

        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.train_transform_crop = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=crop_size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        if self.is_training:
            # if self.attn_head:
            #    return (
            #        self.train_transform(x),
            #        self.train_transform_crop(x),
            #        self.train_transform(x),
            #        self.train_transform_crop(x),
            #    )
            return self.train_transform(x), self.train_transform_crop(x)
        else:
            return self.test_transform(x)

