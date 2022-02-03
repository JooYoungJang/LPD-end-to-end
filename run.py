# import os
# import torch
# import numpy as np
# import cv2
# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')

# import sys
# sys.path.append(".")

# from ssd_two_stage_end2end import build_ssd
# import argparse

# parser = argparse.ArgumentParser(
#     description='Single Shot MultiBox Detector Testing With Pytorch')
# parser.add_argument('--input_size', default=512, type=int, help='SSD300 or SSD512')
# parser.add_argument('--input_size_2', default=56, type=int, help='input size of the second network')
# parser.add_argument('--expand_num', default=3, type=int, help='expand ratio around the license plate')
# args = parser.parse_args()

# net = build_ssd('test', args.input_size, args.input_size_2, 2, args.expand_num)    # initialize SSD

# # --------------------720p---------------------------

# net.load_weights("/workspace/LPD-end-to-end/weights/ssd512_720p.pth")

# # matplotlib inline
# from matplotlib import pyplot as plt
# from data import CAR_CARPLATE_TWO_STAGE_END2ENDDetection, CAR_CARPLATE_TWO_STAGE_END2ENDAnnotationTransform
# CAR_CARPLATE_TWO_STAGE_END2END_ROOT = "/workspace/LPD-end-to-end/images/720p/"
# testset = CAR_CARPLATE_TWO_STAGE_END2ENDDetection(CAR_CARPLATE_TWO_STAGE_END2END_ROOT, None, None, CAR_CARPLATE_TWO_STAGE_END2ENDAnnotationTransform(),
#                                        dataset_name='test')
# for img_id in range(4):
#     image = testset.pull_image(img_id)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     x = cv2.resize(image, (args.input_size, args.input_size)).astype(np.float32)
#     x -= (104.0, 117.0, 123.0)
#     x = x.astype(np.float32)
#     x = x[:, :, ::-1].copy()
#     x = torch.from_numpy(x).permute(2, 0, 1)

#     xx = x.unsqueeze(0)     # wrap tensor in Variable
#     if torch.cuda.is_available():
#         xx = xx.cuda()

#     detections = net(xx, [])

#     from data import CAR_CARPLATE_TWO_STAGE_END2END_CLASSES as labels

#     fig = plt.figure(figsize=(10, 10))
#     colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
#     plt.imshow(rgb_image)  # plot the image for matplotlib
#     currentAxis = plt.gca()

#     # [num, num_classes, num_car, 10]
#     # 10: score(1) bbox(4) has_lp(1) size_lp(2) offset(2)
#     detections = detections.data
#     # scale each detection back up to the image
#     scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
#     scale_4 = torch.Tensor(rgb_image.shape[1::-1]).repeat(4)

#     for i in range(detections.size(1)):
#         # skip background
#         if i == 0:
#             continue
#         th = 0.6
#         for j in range(detections.size(2)):
#             if detections[0, i, j, 0] > th:
#                 label_name = labels[i-1]
#                 display_txt = '%s: %.2f' % (label_name, detections[0, i, j, 0])
#                 pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
#                 coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
#                 color = colors[i]
                
#                 if i == 2:
#                     lp_pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
#                     lp_coords = (lp_pt[0], lp_pt[1]), lp_pt[2] - lp_pt[0] + 1, lp_pt[3] - lp_pt[1] + 1
#                     four_corners = (detections[0, i, j, 5:]*scale_4).cpu().numpy()
#                     corners_x = np.append(four_corners[0::2], four_corners[0])
#                     corners_y = np.append(four_corners[1::2], four_corners[1])
#                     currentAxis.plot(corners_x, corners_y, linewidth=2, color=colors[0])

#     if not os.path.isdir("/workspace/LPD-end-to-end/results"):
#         os.mkdir("/workspace/LPD-end-to-end/results")
#     plt.savefig(os.path.join("/workspace/LPD-end-to-end/results", "720p_"+str(img_id)+".svg"), bbox_inches='tight')


# # --------------------1080p---------------------------

# net.load_weights("/workspace/LPD-end-to-end/weights/ssd512_1080p.pth")

# # matplotlib inline
# from matplotlib import pyplot as plt
# from data import CAR_CARPLATE_TWO_STAGE_END2ENDDetection, CAR_CARPLATE_TWO_STAGE_END2ENDAnnotationTransform
# CAR_CARPLATE_TWO_STAGE_END2END_ROOT = "/workspace/LPD-end-to-end/images/1080p/"
# testset = CAR_CARPLATE_TWO_STAGE_END2ENDDetection(CAR_CARPLATE_TWO_STAGE_END2END_ROOT, None, None, CAR_CARPLATE_TWO_STAGE_END2ENDAnnotationTransform(),
#                                        dataset_name='test')
# for img_id in range(4):
#     image = testset.pull_image(img_id)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     x = cv2.resize(image, (args.input_size, args.input_size)).astype(np.float32)
#     x -= (104.0, 117.0, 123.0)
#     x = x.astype(np.float32)
#     x = x[:, :, ::-1].copy()
#     x = torch.from_numpy(x).permute(2, 0, 1)

#     xx = x.unsqueeze(0)
#     if torch.cuda.is_available():
#         xx = xx.cuda()

#     detections = net(xx, [])

#     from data import CAR_CARPLATE_TWO_STAGE_END2END_CLASSES as labels

#     fig = plt.figure(figsize=(10, 10))
#     colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
#     plt.imshow(rgb_image)  # plot the image for matplotlib
#     currentAxis = plt.gca()

#     # [num, num_classes, num_car, 10]
#     # 10: score(1) bbox(4) has_lp(1) size_lp(2) offset(2)
#     detections = detections.data
#     # scale each detection back up to the image
#     scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
#     scale_4 = torch.Tensor(rgb_image.shape[1::-1]).repeat(4)

#     for i in range(detections.size(1)):
#         # skip background
#         if i == 0:
#             continue
#         th = 0.6
#         for j in range(detections.size(2)):
#             if detections[0, i, j, 0] > th:
#                 label_name = labels[i-1]
#                 display_txt = '%s: %.2f' % (label_name, detections[0, i, j, 0])
#                 pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
#                 coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
#                 color = colors[i]
                
#                 if i == 2:
#                     lp_pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
#                     lp_coords = (lp_pt[0], lp_pt[1]), lp_pt[2] - lp_pt[0] + 1, lp_pt[3] - lp_pt[1] + 1
#                     four_corners = (detections[0, i, j, 5:]*scale_4).cpu().numpy()
#                     corners_x = np.append(four_corners[0::2], four_corners[0])
#                     corners_y = np.append(four_corners[1::2], four_corners[1])
#                     currentAxis.plot(corners_x, corners_y, linewidth=2, color=colors[0])

#     if not os.path.isdir("/workspace/LPD-end-to-end/results"):
#         os.mkdir("/workspace/LPD-end-to-end/results")
#     plt.savefig(os.path.join("/workspace/LPD-end-to-end/results", "1080p_"+str(img_id)+".svg"), bbox_inches='tight')

from data import *
import os
import torch
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

import sys
sys.path.append(".")

import argparse

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Testing With Pytorch')
parser.add_argument('--input_size', default=512, type=int, help='SSD300 or SSD512')
parser.add_argument('--input_size_2', default=56, type=int, help='input size of the second network')
parser.add_argument('--expand_num', default=3, type=int, help='expand ratio around the license plate')
parser.add_argument('--dataset', default='carplate', type=str, help='dataset type')
args = parser.parse_args()

if args.dataset == 'two_stage':
    from ssd_two_stage_end2end import build_ssd
    net = build_ssd('test', args.input_size, args.input_size_2, 2, args.expand_num)    # initialize SSD
elif args.dataset == 'carplate':
    from ssd import build_ssd
    net = build_ssd('test', args.input_size, 2)



# matplotlib inline
from matplotlib import pyplot as plt
from data import CAR_CARPLATE_TWO_STAGE_END2ENDDetection, CAR_CARPLATE_TWO_STAGE_END2ENDAnnotationTransform, CAR_CARPLATEDetection
CAR_CARPLATE_TWO_STAGE_END2END_ROOT = "/data/ocr/dyiot_lpd"
# CAR_CARPLATE_TWO_STAGE_END2END_ROOT = "/workspace/LPD-end-to-end/images/1080p/"
if args.dataset == 'two_stage':
    testset = CAR_CARPLATE_TWO_STAGE_END2ENDDetection(CAR_CARPLATE_TWO_STAGE_END2END_ROOT, None, None, CAR_CARPLATE_TWO_STAGE_END2ENDAnnotationTransform(),
                                       dataset_name='test')
elif args.dataset == 'carplate':
    testset= CAR_CARPLATEDetection(root=CAR_CARPLATE_TWO_STAGE_END2END_ROOT,
                           transform=BaseTransform(args.input_size, MEANS),
                           target_transform=CAR_CARPLATEAnnotationTransform(keep_difficult=True),
                           dataset_name='test')       

# net.load_weights("/workspace/LPD-end-to-end/weights/ssd512_1080p.pth")
# net.load_weights("/workspace/LPD-end-to-end/weights/ssd512_720p.pth")
net.load_weights("weights/ssd.pytorch/weights/CCPD_carplate_bbox_weights_12345789_500ssd512_25000.pth")

for img_id in range(4840):
    image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (args.input_size, args.input_size)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = x.unsqueeze(0)
    if torch.cuda.is_available():
        xx = xx.cuda()
    if args.dataset == 'two_stage':
        detections = net(xx, [])
        from data import CAR_CARPLATE_TWO_STAGE_END2END_CLASSES as labels
    elif args.dataset == 'carplate':
        detections = net(xx)
        from data import CARPLATE_CLASSES as labels    
        import matplotlib.patches as patches
    
    fig = plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    # [num, num_classes, num_car, 10] -> [1, 3, 200, 13]
        ## num_classes: car, car_plate
        
    # 10: score(1) bbox(4) has_lp(1) size_lp(2) offset(2)
    detections = detections.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    scale_4 = torch.Tensor(rgb_image.shape[1::-1]).repeat(4)

    for i in range(detections.size(1)):
        # skip background
        if i == 0:
            continue
        th = 0.6
        for j in range(detections.size(2)):
            if args.dataset == 'two_stage':
                if i==2:
                    print("conf: {}\n".format(detections[0, i, j, 0]))
            
            if detections[0, i, j, 0] > th:
                print("conf: {}\n".format(detections[0, i, j, 0]))
                label_name = labels[i-1]
                display_txt = '%s: %.2f' % (label_name, detections[0, i, j, 0])
                pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                color = colors[i]
                if args.dataset == 'two_stage':
                    if i == 2:
                        lp_pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
                        lp_coords = (lp_pt[0], lp_pt[1]), lp_pt[2] - lp_pt[0] + 1, lp_pt[3] - lp_pt[1] + 1
                        four_corners = (detections[0, i, j, 5:]*scale_4).cpu().numpy()
                        corners_x = np.append(four_corners[0::2], four_corners[0])
                        corners_y = np.append(four_corners[1::2], four_corners[1])
                        currentAxis.plot(corners_x, corners_y, linewidth=2, color=colors[0])
                elif args.dataset == 'carplate':
                    if i==1:
                        lp_pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
                        # corners_x = lp_pt[0::2]
                        # corners_y = lp_pt[1::2]
                        currentAxis.add_patch(patches.Rectangle((lp_pt[0],lp_pt[1]), lp_pt[2] - lp_pt[0] + 1, lp_pt[3] - lp_pt[1],
                        linewidth=2, edgecolor='r', facecolor='none'))
                        # currentAxis.plot(corners_x, corners_y, linewidth=2, color=colors[0])
    if not os.path.isdir("/workspace/LPD-end-to-end/results_dyiot"):
        os.mkdir("/workspace/LPD-end-to-end/results_dyiot")
    plt.savefig(os.path.join("/workspace/LPD-end-to-end/results_dyiot", "1080p_"+str(img_id)+".svg"), bbox_inches='tight')