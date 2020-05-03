# Core Author: Zylo117
# Script's Author: winter2897 

"""
Simple Inference Script of EfficientDet-Pytorch for detecting objects on webcam
"""
import time
import torch
import cv2
import os
import numpy as np
from torch.backends import cudnn
import sys
sys.path.append('Yet-Another-EfficientDet-Pytorch')
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video

model_list = [0,1,2,3]
force_input_size_list = [ 512, 640, 768, 896, 1024]
videos_list = [ "3.mp4", "4.mp4"]
cwd = os.getcwd()

for model_val in model_list:

    for fis in force_input_size_list:

        dir_name = "EfficientDet D" + str(model_val) + " Input Size " +  str(fis)
        dir = os.path.join(cwd, dir_name)
        try:
        	os.mkdir(dir)
       	except:
       		pass

        for vid in videos_list:

            # Video's path
            video_src = vid  # set int to use webcam, set str to read from a video file

            compound_coef = model_val
            force_input_size = fis  # set None to use default size

            threshold = 0.2
            iou_threshold = 0.2

            use_cuda = True
            use_float16 = False
            cudnn.fastest = True
            cudnn.benchmark = True

            obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                        'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                        'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                        'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                        'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                        'toothbrush']

            # tf bilinear interpolation is different from any other's, just make do
            input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
            input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

            # load model
            model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
            model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
            model.requires_grad_(False)
            model.eval()

            if use_cuda:
                model = model.cuda()
            if use_float16:
                model = model.half()

            # function for display
            def display(preds, imgs):
                for i in range(len(imgs)):
                    if len(preds[i]['rois']) == 0:
                        continue

                    for j in range(len(preds[i]['rois'])):
                        (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
                        obj = obj_list[preds[i]['class_ids'][j]]

                        if("person" in obj or "sports ball" in obj):
                            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                            score = float(preds[i]['scores'][j])

                            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 0), 1)

                    return imgs[i]
            # Box
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            # Video capture
            cap = cv2.VideoCapture(video_src)

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(fps)
            outp = cv2.VideoWriter(dir + "/" + "output_" + vid, fourcc, fps, (frame_width,frame_height))



            while True:
                ret, frame = cap.read()
                if not ret:
                    break


                # frame preprocessing
                ori_imgs, framed_imgs, framed_metas,t1 = preprocess_video(frame,width=input_size,height=input_size)
                print("Preprocessing time",time.time()-t1)
                if use_cuda:
                    x = torch.stack([fi.cuda() for fi in framed_imgs], 0)
                else:
                    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)


                t1=time.time()
                # model predict
                with torch.no_grad():
                    features, regression, classification, anchors = model(x)

                    out = postprocess(x,
                                    anchors, regression, classification,
                                    regressBoxes, clipBoxes,
                                    threshold, iou_threshold)

                # result


                out = invert_affine(framed_metas, out)
                img_show = display(out, ori_imgs)
                t2 = time.time() - t1
                print("Infeence time",t2)
                current_frame_fps = 1.0/t2

                cv2.putText(img_show, 'FPS: {0:.2f}'.format(current_frame_fps), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0),
                    2, cv2.LINE_AA)

                # print("D",compound_coef,current_frame_fps)


                # show frame by frame
                outp.write(img_show)

            cap.release()
            outp.release()

# # Video's path
# video_src = '3.mp4'  # set int to use webcam, set str to read from a video file
#
# compound_coef = 0
# force_input_size = None  # set None to use default size
#
# threshold = 0.2
# iou_threshold = 0.2
#
# use_cuda = True
# use_float16 = False
# cudnn.fastest = True
# cudnn.benchmark = True
#
# obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#             'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#             'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
#             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#             'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
#             'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
#             'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
#             'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#             'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#             'toothbrush']
#
# # tf bilinear interpolation is different from any other's, just make do
# input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
# input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
#
# # load model
# model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
# model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
# model.requires_grad_(False)
# model.eval()
#
# if use_cuda:
#     model = model.cuda()
# if use_float16:
#     model = model.half()
#
#
# # function for display
# def display(preds, imgs):
#     for i in range(len(imgs)):
#         if len(preds[i]['rois']) == 0:
#             continue
#
#         for j in range(len(preds[i]['rois'])):
#             (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
#             obj = obj_list[preds[i]['class_ids'][j]]
#
#             if ("person" in obj or "sports ball" in obj):
#                 cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
#                 score = float(preds[i]['scores'][j])
#
#                 cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
#                             (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                             (255, 255, 0), 1)
#
#         return imgs[i]
#
#
# # Box
# regressBoxes = BBoxTransform()
# clipBoxes = ClipBoxes()
#
# # Video capture
# cap = cv2.VideoCapture(video_src)
#
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fourcc = cv2.VideoWriter_fourcc(*'MPEG')
# outp = cv2.VideoWriter("output_vid.mp4", fourcc, 30, (frame_width, frame_height))
#
# while True:
#     # t1 = time.time()
#
#     ret, frame = cap.read()
#     if not ret:
#         break
#     t1=time.time()
#     print(frame.shape)
#     # frame preprocessing
#     ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)
#
#     if use_cuda:
#         x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
#     else:
#         x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
#
#     x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
#
#     # model predict
#     with torch.no_grad():
#         features, regression, classification, anchors = model(x)
#
#         out = postprocess(x,
#                           anchors, regression, classification,
#                           regressBoxes, clipBoxes,
#                           threshold, iou_threshold)
#
#     # result
#     t2 = time.time() - t1
#     print(len(framed_metas))
#     out = invert_affine(framed_metas, out)
#     img_show = display(out, ori_imgs)
#
#     current_frame_fps = (1.0 / t2)
#
#     cv2.putText(img_show, 'FPS: {0:.2f}'.format(current_frame_fps), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                 (255, 255, 0),
#                 2, cv2.LINE_AA)
#
#     print("D", compound_coef, current_frame_fps)
#     # cv2.imwrite('a.jpg',img_show)
#     # show frame by frame
#     outp.write(img_show)
#
# cap.release()
# outp.release()