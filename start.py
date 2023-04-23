import sys
import train


def main():
    opt = train.parse_opt();
    opt.resume = True;
    opt.project = "runs/train"
    train.main(opt)


if __name__ == '__main__':
    main()
