import detect
from segment import predict

#摄像头处理
def main():
    opt = detect.parse_opt();
    opt.weights = "best.pt"
    opt.source = 0;
    #opt.resume = True;
    #opt.device = "0"
    detect.main(opt)

if __name__ == '__main__':
    main()