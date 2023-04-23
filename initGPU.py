import torch


def init_gpu():
    torch.Tensor([10, 10, 10]).cuda()


def main():
    init_gpu()


if __name__ == '__main__':
    main()
