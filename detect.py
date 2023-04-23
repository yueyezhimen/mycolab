# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
#https://blog.csdn.net/weixin_43334693/article/details/129349094
#https://www.bilibili.com/video/BV1Dt4y1x7Fz
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    #第一部分 对图片处理的部分
    #'data/images',  # file/dir/URL/glob/screen/0(webcam)
    #转路径为字符串  source == data\\images\\bus.jpg
    source = str(source)
    #save img nosave == false
    # source.endswith('.txt')判断source是否是.txt文件
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    #Path(source).suffix[1:] suffix后缀 [1:] 取出img mp4 等等 判断是否是IMG_FORMATS VID_FORMATS
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    #lower()转小写 是否是http网络链接 是否是url文件？
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    #source是否是数字 也就是摄像头编号 endswith结尾是否是streams流 is_url是否是网络链接 也就是是不是摄像头
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    #是否是屏幕图片
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        #如果是链接且是网络文件则进行下载
        source = check_file(source)  # download

    #第二部分 Directories 新建对结果保存的文件夹
    #project == runs/detect 保存结果的路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #是否保存结果的文件txt save_txt=true 则（save_dir/'labels'）.mkdir创建目录 也就是runs/detect/exp5/labels创建目录
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    #第三部分 Load model 加载模型的权重 cuda device device==0 cuda 0
    device = select_device(device)
    #Multi 多个 Backend后端 多个模型，利用weights的最后判断使用哪个框架，PyTorch TorchScript ONNX TensorFlow
    # 指数据集配置文件的路径
    '''
    weights 指模型的权重路径 
    device 指设备
    dnn 指是否使用OpenCV
    data 指数据集配置文件的路径 例如data/coco.yaml
    fp16 指是否使用半精度浮点数进行推理
    '''
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    '''
        stride：推理时所用到的步长，默认为32， 大步长适合于大目标，小步长适合于小目标
        names：保存推理结果名的列表，比如默认模型的值是['person', 'bicycle', 'car', ...] 
        pt: 加载的是否是pytorch模型（也就是pt格式的文件）
        jit：当某段代码即将第一次被执行时进行编译，因而叫“即时编译”
        onnx：利用Pytorch我们可以将model.pt转化为model.onnx格式的权重，在这里onnx充当一个后缀名称，
              model.onnx就代表ONNX格式的权重文件，这个权重文件不仅包含了权重值，也包含了神经网络的网络流动信息以及每一层网络的输入输出信息和一些其他的辅助信息。
    '''
    #model.stride 模型步长 model.names 所有类别名 model.pt 是否是pytorch
    stride, names, pt = model.stride, model.names, model.pt
    #检查图片imgsz保证图片尺寸为32的倍数，不是则自动计算出32倍数尺寸
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    bs = 1  # batch_size 每次喂给模型1张图片，预测阶段。不需要一次喂多个
    # 第二部分 Dataloader 定义了个 Dataloader 模块 加载带预测的图片
    if webcam:
        #摄像头
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        #截屏
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        #图片
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    #bs == 2 [None, None]
    vid_path, vid_writer = [None] * bs, [None] * bs

    #第五部分 Run inference 执行模型的推理过程 并将检测框画出来
    #热身 传入一个空白图片 调用GPU进行热身
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    #存储结果信息
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    #     self.img_size = img_size
    #     self.stride = stride
    #     #将视频和图片都合并起来
    #     self.files = images + videos
    #     self.nf = ni + nv  # number of files
    #     #视频标志 [False] * ni ni 个false nv个true 然后拼接
    #     self.video_flag = [False] * ni + [True] * nv
    #     self.mode = 'image'
    #     self.auto = auto
    #     self.transforms = transforms  # optional
    #     self.vid_stride
    # py中的特性 遍历dataset 遍历所有图片 中的迭代器  __iter__(self)函数 后__next__(self)函数
    for path, im, im0s, vid_cap, s in dataset:
        # path图片路径 im是变换后的图片CHW RGB  im0是原图 s为输出的信息

        #图片预处理 with 是执行 __enter__ 结束执行__exit__ 用于计时 Profile()
        with dt[0]:
            #im类型为numpy 转为Tersor类型 并且设置将图片tensor转入到model.device这个GPU中
            im = torch.from_numpy(im).to(model.device)
            #半精度
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            #将图片归一化
            im /= 255  # 0 - 255 to 0.0 - 1.0
            #判断图片是不是只有3
            if len(im.shape) == 3:
                #是的话将图片扩充 expand for batch dim 添加一个第0维。缺少batch这个尺寸，所以将它扩充一下，变成[1，3,640,48
                im = im[None]  # expand for batch dim 增加一个维度，把batchsize个所有的数据都塞入矩阵中

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            #visualize是否保存特征图，augment是否做数据增强
            #[1,18900,85]
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        #[1,5,6,calss]
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        #第六部分 Process predictions 处理打印信息
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    #--weights：  训练的权重路径，可以使用自己训练的权重，也可以使用官网提供的权重。默认官网的权重yolov5s.pt(yolov5n.pt/yolov5s.pt/yolov5m.pt/yolov5l.pt/yolov5x.pt/区别在于网络的宽度和深度以此增加)
    # --source：  测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头)，也可以是rtsp等视频流, 默认data/images
    # --data：  配置数据文件路径，包括image/label/classes等信息，训练自己的文件，需要作相应更改，可以不用管
    # --imgsz：  预测时网络输入图片的尺寸，默认值为 [640]
    # --conf-thres：  置信度阈值，默认为 0.50
    # --iou-thres：  非极大抑制时的 IoU 阈值，默认为 0.45
    # --max-det：  保留的最大检测框数量，每张图片中检测目标的个数最多为1000类
    # --device：  使用的设备，可以是 cuda 设备的 ID（例如 0、0,1,2,3）或者是 'cpu'，默认为 '0'
    # --view-img：  是否展示预测之后的图片/视频，默认False
    # --save-txt：  是否将预测的框坐标以txt文件形式保存，默认False，使用--save-txt 在路径runs/detect/exp*/labels/*.txt下生成每张图片预测的txt文件
    # --save-conf：  是否保存检测结果的置信度到 txt文件，默认为 False
    # --save-crop：  是否保存裁剪预测框图片，默认为False，使用--save-crop 在runs/detect/exp*/crop/剪切类别文件夹/ 路径下会保存每个接下来的目标
    # --nosave：  不保存图片、视频，要保存图片，不设置--nosave 在runs/detect/exp*/会出现预测的结果
    # --classes：  仅检测指定类别，默认为 None
    # --agnostic-nms：  是否使用类别不敏感的非极大抑制（即不考虑类别信息），默认为 False
    # --augment：  是否使用数据增强进行推理，默认为 False
    # --visualize：  是否可视化特征图，默认为 False
    # --update：  如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    # --project：  结果保存的项目目录路径，默认为 'ROOT/runs/detect'
    # --name：  结果保存的子目录名称，默认为 'exp'
    # --exist-ok：  是否覆盖已有结果，默认为 False
    # --line-thickness：  画 bounding box 时的线条宽度，默认为 3
    # --hide-labels：  是否隐藏标签信息，默认为 False
    # --hide-conf：  是否隐藏置信度信息，默认为 False
    # --half：  是否使用 FP16 半精度进行推理，默认为 False
    # --dnn：  是否使用 OpenCV DNN 进行 ONNX 推理，默认为 False
    # ————————————————
    # 版权声明：本文为CSDN博主「路人贾'ω'」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/weixin_43334693/article/details/129349094
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    #640*640
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    #检测包是否都有 ---requirements.txt
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
