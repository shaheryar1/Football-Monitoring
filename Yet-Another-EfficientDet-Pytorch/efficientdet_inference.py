import time
import torch
import cv2
import os
import numpy as np
from torch.backends import cudnn
import sys
from Utils.Color import extract_color
from Utils.ImageUtitls import drawBoxes,convert_bbox_to_deep_sort_format
sys.path.append('Yet-Another-EfficientDet-Pytorch')

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video



from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort.detection import Detection as ddet

# Constants
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
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]


color_dict={}

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

def decode_predictions(preds,classes_filter = ["person"]):
    boxes= []
    classes=[]
    scores=[]
    for j in range(len(preds["rois"])):
        obj = obj_list[preds['class_ids'][j]]
        if (obj in classes_filter):
            score = float(preds['scores'][j])
            (x1, y1, x2, y2) = preds['rois'][j].astype(np.int)
            boxes.append((x1, y1, x2, y2))
            scores.append(score)
            classes.append(obj)
    return boxes,classes,scores


def update_color_association(roi, id):
    if(color_dict.get(id) is not  None):
        return color_dict[id]
    else:
        H, W, C = roi.shape
        roi = roi[0:int(H / 2), :]
        if(H<=10 or W <=10):
            return [0,0,0]
        name,color_range = extract_color(roi, visualize=False)
        color_dict[id]=color_range[1]
        return color_dict[id]




def efficientDet_video_inference(video_src,compound_coef = 0,force_input_size=None,
                                 frame_skipping = 3,
                                 threshold=0.2,out_path=None,imshow=False,
                                 display_fps=False):

    #deep-sort variables

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0


    model_filename = '/home/shaheryar/Desktop/Projects/Football-Monitoring/deep_sort/model_weights/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric,n_init=5)

    # efficientDet-pytorch variables
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
        if (frame_skipping==0 or i%frame_skipping==0):
        # if(True):


            # frame preprocessing (running detections)
            ori_imgs, framed_imgs, framed_metas, t1 = preprocess_video(frame, width=input_size, height=input_size)
            if use_cuda:
                x = torch.stack([fi.cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
            # model predict
            t1=time.time()
            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  threshold, iou_threshold)
            # Post processing
            out = invert_affine(framed_metas, out)
            # decoding bbox ,object name and scores
            boxes,classes,scores =decode_predictions(out[0])
            org_boxes = boxes.copy()
            t2 = time.time() - t1

            # feature extraction for deep sort
            boxes = [convert_bbox_to_deep_sort_format(frame.shape, b) for b in boxes]

            features = encoder(frame,boxes)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
            boxes = np.array([d.tlwh for d in detections])
            # print(boxes)
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            tracker.predict()
            tracker.update(detections)



        i = i + 1
        img_show=frame.copy()
        for j in range(len(org_boxes)):
            img_show =drawBoxes(img_show,org_boxes[j],(255,255,0),str(tracker.tracks[j].track_id))

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            x1=int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2=int(bbox[3])
            roi= frame[y1:y2,x1:x2]
            cv2.rectangle(img_show, (x1, y1), (x2, y2), update_color_association(roi, track.track_id), 2)
            cv2.putText(img_show, str(track.track_id), (x1, y1), 0, 5e-3 * 100, (255, 255, 0), 1)


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
            # print(color_dict)

        if imshow:
            img_show=cv2.resize(img_show,(0,0),fx=0.75,fy=0.75)
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
    efficientDet_video_inference(video_path,compound_coef=0,force_input_size=768,
                                 out_path=None
                                 ,imshow=True,frame_skipping=5,display_fps=True)
