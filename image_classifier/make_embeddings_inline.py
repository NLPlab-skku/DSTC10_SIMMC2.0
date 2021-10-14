import json
import os
from os import listdir
from os.path import isfile, join

######## Get pre-trained from detectron ##################
import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# import some common libraries
import numpy as np
import cv2
import torch

from torch import nn

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

# FOLDER = os.path.dirname(os.path.abspath(__file__))+"/"
FOLDER = os.path.dirname(os.path.abspath(__file__))+"/"
DET_FOLDER = "../py-bottom-up-attention/"
print(FOLDER)
print(os.path.abspath(os.getcwd()))

# with open(FOLDER + "config/faster_rcnn_R_101_C4_caffemaxpool.yaml","r") as f:
#   print (f.read())
cfg = get_cfg()
cfg.merge_from_file(DET_FOLDER + "configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
# cfg.merge_from_file(DET_FOLDER + "configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
# VG Weight
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
predictor = DefaultPredictor(cfg)

###############################

def load_json(filename):
    with open(filename,mode="r") as f:
        return json.load(f)

def save_json(filename,data):
    with open(filename, mode="w") as f:
        json.dump(data,f)
    return


def doit(raw_image, raw_boxes):
        # Process Boxes
    raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())
    
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (raw_height, raw_width))
        
        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        print("Transformed image size: ", image.shape[:2])
        
        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        #print(scale_x, scale_y)
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)
        
        # ----
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        print('Pooled features size:', feature_pooled.shape)
        
        # Predict classes        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled) and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        pred_class_prob = nn.functional.softmax(pred_class_logits, -1)
        pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)
        
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)
        
        # Detectron2 Formatting (for visualization only)
        roi_features = feature_pooled
        instances = Instances(
            image_size=(raw_height, raw_width),
            pred_boxes=raw_boxes,
            scores=pred_scores,
            pred_classes=pred_classes,
            attr_scores = max_attr_prob,
            attr_classes = max_attr_label
        )
        
        return instances, roi_features
    
    
class D2input:
    def __init__(self,filename):
        #data에 전부 저장.
        self.data=[]
        bboxfolder=FOLDER + "simmc2_scene_jsons_dstc10_public/"
        image_path1=FOLDER +"simmc2_scene_images_dstc10_public_part2"
        image_path2=FOLDER +"simmc2_scene_images_dstc10_public_part1"
        image_files1 = [f for f in listdir(image_path1) if isfile(join(image_path1, f))]
        image_files2 = [f for f in listdir(image_path2) if isfile(join(image_path2, f))]
        
#         im = cv2.imread("data/images/cloth_store_1416238_woman_9_2.png")

        dialogue_data=load_json(filename)
        
        for ind,d in enumerate(dialogue_data["dialogue_data"]):
            try:
                tmp_d = d
                d["features"]={}
                for key,scene in tmp_d["scene_ids"].items(): #key에 scene이 사용되기 시작하는 turn idx 들어감.
                    bbox_data=load_json(bboxfolder+scene+"_scene.json")
                    objects=bbox_data["scenes"][0]["objects"]
                    objects=[o for o in objects if o["index"] in d["mentioned_object_ids"]]
                  
                    bbox_list=[]
                    object_index_list=[]
                    
                    for i,o in enumerate(objects):
                        bbox_list.append(o["bbox"])
                        object_index_list.append(o["index"])
                        
                    #bbox detectron2 input에 맞게 수정
                    bbox_list=np.array([[d[0],d[1],d[0]+d[3],d[1]+d[2]] for d in bbox_list]) #왼쪽 위 꼭지점과 오른쪽 아래 꼭지점
                    
    #                 tmp_d["object_infos"][key]=objects
                    if scene[0]=="m":
                        scene=scene[2:]
                    if scene + ".png" in image_files1:
                        im = cv2.imread(image_path1 + "/" + scene + ".png")
                    elif scene + ".png" in image_files2:
                        im = cv2.imread(image_path2 + "/" + scene + ".png")
                    else:
                        print("there is no image available")
                    
                    # get embeddings
                    im_height, im_width = im.shape[:2]
                    coords=[
                        [b[0]/im_width,
                        b[1]/im_height,
                        b[2]/im_width,
                        b[3]/im_height,
                        (b[0]+b[2])/(2*im_width),
                        (b[1]+b[3])/(2*im_height),
                        (b[2]-b[0])/im_width,
                        (b[3]-b[1])/im_height
                    ] for b in bbox_list]

                    instances, features = doit(im, bbox_list)
                    features=features.tolist()
                    new_features=[]
                    for fi,f in enumerate(features): # object embedding size become 2056.
                        f.extend(coords[fi])
                        new_features.append(f)
                    features=new_features
                    print(len(new_features[0]))
                    d["features"][key]={}
                    
                    for j,k in zip(object_index_list,features):
                        d["features"][key][str(j)]=k

                dialogue_data.append(tmp_d)
            except:
              pass
        self.dialogue_data=dialogue_data
    
    def save(self):
        save_filename=FOLDER +"bbox_data.json"
        save_json(save_filename,self.data)
    
    def save_with_add_data(self,filename):
        save_json(filename,self.dialogue_data)

        
if __name__ == "__main__":
    dataset_type="dev" # dev/devtest/train
    D2input(FOLDER +f"simmc2_dials_dstc10_{dataset_type}.json").save_with_add_data(FOLDER +f"{dataset_type}_with_mentioned.json")
        
        