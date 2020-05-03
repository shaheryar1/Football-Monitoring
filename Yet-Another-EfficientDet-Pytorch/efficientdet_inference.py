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
def display(preds, imgs):

    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]


        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]

            if ("person" in obj or "sports ball" in obj):
                cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                score = float(preds[i]['scores'][j])

                cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                            (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1)

        return imgs[i]

def efficientDet_video_inference(video_src,compound_coef = 0,force_input_size=None,
                                 frame_skipping = 3,
                                 threshold=0.2,out_path=None,imshow=False,
                                 display_fps=False):
    iou_threshold = 0.4
    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

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

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # Video capture
    cap = cv2.VideoCapture(video_src)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Video fps",fps)
    if(out_path is not None):
        outp = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
    i=0
    start= time.time()
    current_frame_fps=0
    while True:

        ret, frame = cap.read()

        if not ret:
            break
        t1=time.time()
        if (frame_skipping>0 and i%frame_skipping==0):
            # frame preprocessing (running detections)
            ori_imgs, framed_imgs, framed_metas, t1 = preprocess_video(frame, width=input_size, height=input_size)
            if use_cuda:
                x = torch.stack([fi.cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
            # model predict
            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  threshold, iou_threshold)
            out = invert_affine(framed_metas, out)

        i = i + 1


        img_show = display(out, [frame])
        t2 = time.time() - t1
        if display_fps:
            current_frame_fps=1/t2
        else:
            current_frame_fps=0

        cv2.putText(img_show, 'FPS: {0:.2f}'.format(current_frame_fps), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0),
                    2, cv2.LINE_AA)
        if (i % int(fps) == 0):
            print("Processed ", str(int(i / fps)), "seconds")
            print("Time taken",time.time()-start)

        if imshow:
            cv2.imshow('Frame',img_show)
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if out_path is not None:
            outp.write(img_show)

    cap.release()
    outp.release()

if __name__ == '__main__':
    video_path = "3.mp4"
    efficientDet_video_inference(video_path,compound_coef=0,force_input_size=768,out_path=None,imshow=True,frame_skipping=6)
