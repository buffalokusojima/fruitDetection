import sys
sys.path.append('../')
from src import kerasYolo3

import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os, glob, csv, sys

import argparse

from timeit import default_timer as timer

def get_veriety_yolo_map(folder):
    
    yolo_map = {}
    
    """
    Here shoud be the code which puts models into 'yolo_map' from specific folders
    , each of which has model files 
    
    written below
    """
    for path in glob.glob(os.path.join(folder, '*')):
        name = os.path.basename(path)
        name = name.split("_")[-1]
        model_file = os.path.join(path, 'veriety-model.h5')
        class_file = os.path.join(path.replace('model_data', 'class').replace('モデル','クラス'), 'class.txt') 
        
        yolo = kerasYolo3.ImageDetector(model_path=model_file, classes_path=class_file)
        yolo_map[name] = yolo
    
    return yolo_map

def search_bbox(bbox, result_list):
    iou_max = 0
    iou_index = None
    for i, result in enumerate(result_list):
        if result[2] == None:
            continue
        iou =get_iou(bbox, result[1:5])
        if iou > iou_max:
            iou_max = iou
            iou_index = i
    return iou_index

def get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = abs(xB - xA) * abs(yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0])) * abs((boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0])) * abs((boxB[3] - boxB[1]))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def detect_folder(img_folder, type_model, veriety_model_map, label_map, bbox_data_map):
   
    result_list = []
    for path in glob.glob(os.path.join(img_folder, '*')):
        if(path.split(".")[-1] == 'jpg'):
            print("loading image:", path)
            image, result = type_model.detect_image(path)
            if not result or len(result) == 0:
                type_predict = "リンゴ" #裏技
            else:
                index = search_bbox(bbox_data_map[os.path.basename(path)], result)
                type_predict = result[index][0]
            file_name = path.split("/")[-1]
            
            veriety_model = veriety_model_map.get(type_predict)
            
            image, result = veriety_model.detect_image(path)
        
            if not result or len(result) == 0:
                veriety_predict = None
            else:
                index = search_bbox(bbox_data_map[os.path.basename(path)], result)
                veriety_predict = result[index][0] #上記同様
        
            type_result = label_map.get(type_predict)
            if veriety_predict:   
                veriety_result = type_result.get('veriety').get(veriety_predict)
            else:
                veriety_result = type_result.get('veriety').get(list(type_result.get('veriety'))[0])
            print(veriety_result)
            result_list.append([file_name, int(type_result.get('type')),
                                int(veriety_result)])
            
            if len(result_list) % 10 == 0:
                print(len(result_list), "image detected")
    return result_list
        

def get_class(type_file):
    type_list = []
    with open(type_file, 'r', encoding='UTF-8') as f:
        line = f.readline()
        while line:
            type_list.append(line.replace("\n",""))
            line = f.readline()
            
    return type_list


def get_label(label_file):
   
    label_map = {}
    with open(label_file, encoding="UTF-8") as f:
        reader = csv.reader(f)
        for row in reader:
            labelType={}
            if not label_map.get(row[2]):
                labelType = label_map[row[2]] = {'type': row[0]}
            else: 
                labelType =  label_map.get(row[2])

            if not labelType.get("veriety"):
                veriety_map =  labelType['veriety'] = {}
            if not veriety_map.get(row[3]):
                veriety_map[row[3]] =  row[1]
    return label_map

def get_bbox_data(bbox_data_file):
    
    bbox_data_map = {}
    with open(bbox_data_file, encoding="UTF-8") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                bbox_data_map[row[0]] = [int(row[1]), int(row[3]), int(row[2]), int(row[4])]
            except Exception as e:
                print(e)
                
    return bbox_data_map
            
def write_result(output_file, result_list):
    with open(output_file, 'w', encoding='UTF-8', newline="") as f: #newlineは勝手に空白入れられる対策
        writer = csv.writer(f, delimiter=",")
        for result in result_list:
            writer.writerow(result)
            
    print("result dumped at:", output_file)
        

if __name__ is '__main__':
    print('executed with         : ', sys.argv[1:])
    start = timer()
    print("Start time:", start)
    p = argparse.ArgumentParser()
    p.add_argument('img', help='image folder path')
    p.add_argument('bbox', help='bbox file path')
    p.add_argument('label', help='label file path')
    p.add_argument('output', help='output folder path')
    
    args = p.parse_args()
    
    
    img_folder = args.img
    bbox_data_file = args.bbox
    label_file = args.label
    output = args.output
    
    if not os.path.exists(img_folder):
        print(img_folder, "does not exist")
        exit(1)
        
    if not os.path.exists(bbox_data_file):
        print(bbox_data_file, "does not exist")
        exit(1)
    
    if not os.path.exists(label_file):
        print(label_file, "does not exist")
        exit(1)
        
    if not os.path.exists(output):
        print(output, "does not exist")
        exit(1)
    
    output_file = os.path.join(output, "output.csv")
    
    class_file = '../class/種類クラス/class.txt'
    type_model_path = '../model_data/種類モデル/type-model.h5' #種類判定モデル
    veriety_model_folder = '../model_data/品種モデル/' #品種判定モデル
    
    print("loading class...", class_file)
    
    type_class = get_class(class_file)
    label_map = get_label(label_file)
    bbox_data_map = get_bbox_data(bbox_data_file)
    #print(bbox_data_map)
    print(label_map)
    type_model = kerasYolo3.ImageDetector(model_path=type_model_path, classes_path=class_file)
    veriety_model_map = get_veriety_yolo_map(veriety_model_folder)
    
    result_list = detect_folder(img_folder, type_model, veriety_model_map, label_map, bbox_data_map)
    
    write_result(output_file, result_list)
    
    type_model.close_session()
    
    for key in veriety_model_map:
        veriety_model_map.get(key).close_session()
    
    end = timer()
    print("End time:", end)
    print(end - start)