from tkinter import *
import tkinter.filedialog

import torch
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from matplotlib import pyplot as plt
import time
import sys
sys.path.append(".")

from data import CARPLATE_CLASSES as labels
# load model
from ssd import build_ssd
ssd_net = build_ssd('test', 512, 2)    # initialize SSD
# ssd_net.load_weights('weights/ssd.pytorch/weights/CCPD_carplate_bbox_weights_12345789_1ssd512_25000.pth')
ssd_net.load_weights('weights/ssd.pytorch/weights/best_ckpt.pth')
import argparse
from recognition_model import Model, CTCLabelConverter
# crnn params
recognition_model_path = 'ssd.pytorch/weights/TPS-VGG_KOR-BiLSTM-CTC-total-lr0.15-PAD-aihub-h80-w240.pth'
alphabet = '0,1,2,3,4,5,6,7,8,9,rk,sk,ek,fk,ak,qk,tk,dk,wk,rj,sj,ej,fj,aj,qj,tj,dj,wj,rh,sh,eh,fh,ah,qh,th,dh,wh,rn,sn,en,fn,an,qn,tn,dn,wn,gj,gk,gh,d,b,l,k,n,j,f,h,a,세종,i,c,m,o,p,e,g,배,크,임'
nclass = len(alphabet.split(',')) + 1
parser = argparse.ArgumentParser()
parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default='VGG_KOR', help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--imgH', type=int, default=80, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=240, help='the width of the input image')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
parser.add_argument('--num_class', type=int, default=nclass, help='recognition class')
parser.add_argument('--batch_max_length', type=int, default=20, help='maximum-label-length')
parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
parser.add_argument('--PAD', type=bool, default=True, help='whether to keep ratio then pad for image resize')
opt = parser.parse_args()
lp_rec_model = Model(opt)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lp_rec_model = torch.nn.DataParallel(lp_rec_model).to(device)
lp_rec_model.load_state_dict(torch.load(recognition_model_path))

# lpr 인식 결과 디코딩 
# lpr 텍스트 정보 인식 
def lpr_recognition(cropped_image, model):
    """Decode encoded texts back into strs.

        Args:
            numpy: a RGB license plate image
            model: <class 'models.crnn_fc.CRNN_FC'>

        Returns:
            str: predict result
    """

    imgH = 80
    imgW = 240
    mean = 0.5
    std = 0.5
    converter = CTCLabelConverter(alphabet)
    # converter.multi_character = False
    # print(alphabet)
    image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    ### ratio
    ### 280은 중국어 훈련 세트의 이미지 너비이고 160은 원본 이미지를 축소한 후의 이미지 너비입니다. 
    w_now = int(image.shape[1] / (100 * 1.0 / imgW))
    h, w = image.shape
    image = cv2.resize(image, (0,0), fx=imgW/w, fy=imgH/h, interpolation=cv2.INTER_CUBIC)

    image = (np.reshape(image, (imgH, imgW, 1))).transpose(2, 0, 1)

    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.sub_(mean).div_(std)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = torch.autograd.Variable(image)

    model.eval()
    text_for_pred = torch.LongTensor(1,opt.batch_max_length + 1).fill_(0).to(device)
    # text, length = converter.encode(text_for_pred, batch_max_length=opt.batch_max_length)
    preds = model(image, text_for_pred).log_softmax(2)
    preds_size = torch.IntTensor([preds.size(1)] * 1)
    preds = preds.permute(1, 0, 2)  # to use CTCloss format
    # Select max probabilty (greedy decoding) then decode index to character
    _, preds_index = preds.max(2)
    preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
    preds_str = converter.decode(preds_index.data, preds_size.data)
    
    print('results: {0}'.format(preds_str))

    return preds_str[0]


from PIL import ImageFont, ImageDraw, Image

font_path = 'ssd.pytorch/recognition_model/NanumPen.ttf'


def putText(img, text, org, font_path, color=(0, 0, 255), font_size=20):
    """
  
    :param img: 
    :param text: 표시할 텍스트 
    :param org: 
    :param font_path: 
    :param color: , (B,G,R)
    :return:
    """
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    b, g, r = color
    a = 0
    draw.text(org, text, stroke_width=2, font=ImageFont.truetype(font_path, font_size), fill=(b, g, r, a))
    img = np.array(img_pil)
    return img


def video_run(dir_name):
    videoCapture = cv2.VideoCapture(dir_name)
    video_name = dir_name.strip().split('/')[-1].split('.')[0]
    video_suffix = dir_name.strip().split('/')[-1].split('.')[1]
    # from n-th frame
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    success, image = videoCapture.read()
    video = cv2.VideoWriter(video_name+'_result.'+video_suffix, cv2.VideoWriter_fourcc(*'XVID'), 25, size)

    cur_num = 0
 
    while success:
        image_copy = image.copy()
        img_h, img_w, _ = image.shape
        # skip frames
        if cur_num % 1 == 0:
            x = cv2.resize(image, (512, 512)).astype(np.float32)
            x -= (104.0, 117.0, 123.0)
            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()
            x = torch.from_numpy(x).permute(2, 0, 1)

            xx = torch.autograd.Variable(x.unsqueeze(0))     # wrap tensor in Variable
            if torch.cuda.is_available():
                xx = xx.cuda()

            y = ssd_net(xx)

            # [num, num_classes, top_k, 10]
            # 10: score(1) bbox(4) has_lp(1) size_lp(2) offset(2)
            detections = y.data
            
            th = 0.6
            for i in range(1, detections.size(1)):
                # skip background  
                has_lp_idx = detections[0, i, :, 0] > th

                dets = detections[0, i, has_lp_idx, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:5]
                boxes[:, 0] *= img_w
                boxes[:, 2] *= img_w
                boxes[:, 1] *= img_h
                boxes[:, 3] *= img_h
                for q in range(boxes.shape[0]):
                    carplate_ymin = boxes[q, 1]
                    carplate_xmin = boxes[q, 0]
                    carplate_ymax = boxes[q, 3]
                    carplate_xmax = boxes[q, 2]
                    scores = dets[:, 0].cpu().numpy()   
                
                    img_crop = image[int(carplate_ymin):int(carplate_ymax)+1, int(carplate_xmin):int(carplate_xmax)+1]
                    # cv2.namedWindow("img_crop", 0)
                    # cv2.imshow('img_crop', img_crop)
                    # cv2.waitKey(0)

                    predict = lpr_recognition(img_crop, lp_rec_model)
                    # if carplate_ymax - carplate_ymin >= 20:
                    image_copy = cv2.rectangle(image_copy, (int(carplate_xmin), int(carplate_ymin)),
                     (int(carplate_xmax), int(carplate_ymax)), color=(0,0,255), thickness=2)
                    image_copy = putText(image_copy, predict, (carplate_xmin-100, carplate_ymin-100),
                                            font_path, (0, 0, 255), 100)


        video.write(image_copy)
        # cv2.imshow('image', image_copy)
        success, image = videoCapture.read()
        cur_num = cur_num + 1

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    video.release()
    videoCapture.release()
    # cv2.destroyAllWindows()

    return cur_num

import glob, os
# test_video_path = '/data/ocr/dyiot_raw/dyiot_video'
# test_video_path ='/data/ocr/dyiot_raw/대영_차번검토_동영상/차번인식-오탐영상/자릿수오류'
# l_v = glob.glob(os.path.join(test_video_path, "*.mp4"))
test_video_path = '/data/ocr/dyiot_raw/대영_차번검토_동영상'
l_v = glob.glob(os.path.join(test_video_path, "*.mp4"))
for v in l_v:
    begin = time.time()
    cur_num = video_run(v)
    end=time.time()
    print('totol_time:', str(end-begin))
    print('totol_frame:', str(cur_num))
    print('FPS:', str(int(cur_num/(end-begin))))


# select video from Dialog
# root = Tk()


# def xz():
#     filename = tkinter.filedialog.askopenfilename()
#     if filename != '':
#         lb.config(text="선택한 동영상은 다음과 같습니다.: "+filename)
#         begin = time.time()
#         cur_num = video_run(filename)
#         end=time.time()
#         print('totol_time:', str(end-begin))
#         print('totol_frame:', str(cur_num))
#         print('FPS:', str(int(cur_num/(end-begin))))
#     else:
#         lb.config(text="선택한 동영상이 없습니다. ")

# lb = tkinter.Label(root, text='동영상 파일을 선택하세요 ')
# lb.pack()
# btn = tkinter.Button(root, text="비디오 선택 대화 상자가 나타납니다.", command=xz)
# btn.pack()
# root.mainloop()
