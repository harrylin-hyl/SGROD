# coding:utf-8
import os
import os.path
import xml.dom.minidom
import xml.etree.ElementTree as et
import cv2
from PIL import Image
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np
import cv2
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
from copy import deepcopy

def box_xywh_to_xyxy(box_xywh: torch.Tensor) -> torch.Tensor:
    box_xyxy = deepcopy(box_xywh)
    box_xyxy[2] = box_xyxy[2] + box_xyxy[0]
    box_xyxy[3] = box_xyxy[3] + box_xyxy[1]
    return box_xyxy
    
root_path = './data/OWOD/'
Annotations_path = root_path + 'Annotations/'
w_Annotations_path = root_path + 'Annotations_sam'


split_f = root_path + 'ImageSets/train.txt'
with open(os.path.join(split_f), "r") as f:
    files = [x.strip() for x in f.readlines()]
# files = os.listdir(Annotations_path)

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to("cuda")
mask_generator = SamAutomaticMaskGenerator(model=sam,
    points_per_side=32,
    # pred_iou_thresh=0.88,
    # stability_score_thresh=0.95,
    pred_iou_thresh=0.95,
    stability_score_thresh=0.95,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=200,  # Requires open-cv to run post-processing)
)

for xmlFile in files:  
    if not os.path.exists(w_Annotations_path + '/' + xmlFile + '.xml'):  
        # label = xmlFile.strip('.xml')
        # xmlFile = label
        label = xmlFile
        image_path = root_path + 'JPEGImages' + '/' + str(label) + '.jpg'

        img = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]

        doc = et.parse(os.path.join(Annotations_path, xmlFile + ".xml"))

        root_doc = doc.getroot()
        if image.shape[-1] != 3:
            doc.write(root_path + 'Annotations_sam' + '/' + xmlFile)
            print(xmlFile, 'error image')
            continue

        masks = mask_generator.generate(image)
        boxes = [box_xywh_to_xyxy(mask["bbox"]) for mask in masks]
        search_region = {}
        i = 0
        for box in boxes:

            search_region['segment_region_' + str(i)] = (box[0], box[1], box[2], box[3])
            i = i + 1
      
        ns = et.SubElement(root_doc, 'segment_region', attrib={})
        for j in range(len(search_region)):

           
            nb = et.SubElement(ns, 'bndbox', attrib={})
            nxmin = et.SubElement(nb, 'xmin', attrib={})
            nxmin.text = str(search_region['segment_region_' + str(j)][0])

            nymin = et.SubElement(nb, 'ymin', attrib={})
            nymin.text = str(search_region['segment_region_' + str(j)][1])

            nxmax = et.SubElement(nb, 'xmax', attrib={})
            nxmax.text = str(search_region['segment_region_' + str(j)][2])

            nymax = et.SubElement(nb, 'ymax', attrib={})
            nymax.text = str(search_region['segment_region_' + str(j)][3])

            et.dump(ns)
        doc.write(root_path + 'Annotations_sam' + '/' + xmlFile + ".xml")
        print(xmlFile,'OK')
