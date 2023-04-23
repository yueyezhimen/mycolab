import sys
import train
import detect
def _detect():
    opt = train.parse_opt();
    opt.weights = ""
    train.main(opt)
def _train():
    opt = train.parse_opt();
    opt.data = "VOC.yaml";
    opt.cfg = "yolov5l.yaml"
    opt.weights = ""
    train.main(opt)
def _resume():
    opt = train.parse_opt();
    opt.resume = True;
    opt.project = "runs/train"

    train.main(opt)

def main():
    _train()


if __name__ == '__main__':
    main()
