"""
8/5 指摘をいくつか追記しました。参考にどうぞ(深津)
"""


import src.kerasYolo3
from src.kerasYolo3 import util
from src.kerasYolo3.image_detection import ImageDetector


def search_bbox(bbox, result_list):
    iou_max = 0
    iou_index = None
    for i, result in enumerate(result_list):
        if result[2] == None:
            continue
        iou =get_iou(bbox, result[2:])
        
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
        
        if iou > iou_max:
            iou_max = iou
            iou_index = i
    return iou_index

# return x_test, y_test, bbox_list
def get_test_list(test_file):
    
    x_test = []
    y_test = []
    bbox_list = []
    with open(test_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.split(" ")
            x_test.append(line[0])
            y_test.append(int(line[1].split(",")[-1].replace("\n", "")))
            bbox_list.append([int(num.replace("\n", "")) for num in line[1].split(",")[:-1]])
            line = f.readline()

    return x_test, y_test, bbox_list



def start_test(model_path, classes_path, test_file, outputPath):
    
    #check if each of given files exists
    if not util.check_file_exist(model_path, classes_path, test_file):
        print("path does not exist")
        return
    
    yolo = ImageDetector(model_path, classes_path)
    
    x_test, y_test, bbox_list = get_test_list(test_file)
    
    class_list = util.get_class_list(classes_path)
    if class_list is None:
        print("class list is None")
        return
    
    class_list.append('failed')
    print("use the class list below\n",class_list)
    
    
    """
    write code for test here
    """

"""
(8/5追記)
共通関数でいいかも
"""
#get class list from class_path
def get_class_list(class_path)
    class_list = []
    with open(class_path, 'r') as f:
        line = f.readline()
        while line:
            class_list.append(line.replace("\n",""))
            line = f.readline()
    class_list.append('failed')
    return class_list



# 画像認識回してく、のやつ
# 名前なんて付ければいいからわかんないからとりあえずget_result_list
# TODO:関数名修正
# test_x, class_listを返す関数
# get result_list from test_x and class_list
# param
# test_x: 画像のパスのlist
# class_list: クラスの名前が入ってるlist
def get_result_list(test_x, class_list):
    result_list = []
    for path in test_x:
        image, result = detect_img(path)
        if not result or len(result) == 0:
            result = [[class_list.index(class_list[-1]), None, None, None, None, None]]
        else:
            for r in result:
                r[0] = class_list.index(r[0])
        result_list.append({
            "image": image,
            "result": result
        })

    return result_list

"""
(8/5追記)
共通関数に入れてもいいかも
パスを渡してそのパスが存在しなかったら作成するメソッド
"""
def make_result_directory(test_file, outputPath):

    timestamp = datetime.datetime.now()
    timestamp = str(timestamp).replace(" ", "_").replace(":", "_")
    outputPath = os.path.join(outputPath, timestamp)
    os.mkdir(outputPath)
    print("folder made:", outputPath)

    output_test_file = os.path.join(outputPath, test_file.split("\\")[-1])
    shutil.copyfile(test_file, output_test_file)
    print("copy", test_file, "to", output_test_file)


def get_predict_list(result_list, bbox_list):
    predict_list = []
    for i, result in enumerate(result_list):
        
        index = search_bbox(bbox_list[i], result['result'])
        """
        (8/5追記)
        ここの分岐はいらないかもしれない
        要確認
        分岐消してエラーなければOK
        クラスリストに対するindexがないことはないはず
        """
        if index is None:
            predict_list.append(result['result'][0][0])
        else:
            predict_list.append(result['result'][index][0])
    print(predict_list)

    return predict_list


def save_failed_data(outputPath, predict_list, test_y, test_x, result_list):
    falsePath = os.path.join(outputPath, "False")
    if not os.path.isdir(falsePath):
        os.mkdir(falsePath)
    print("folder made:", falsePath)
    for i, predict in enumerate(predict_list):
        
        if not predict == test_y[i]:
           
             
            """
            (8/5追記)
            共通関数2種類呼び出しで簡略化出来そう
            
            ・フォルダ作成関数
            
            ・ファイル書き込み関数(yolo認識結果ファイルに書き込む)
            
            """
        
            destPath = os.path.join(falsePath, class_list[test_y[i]])
            if not os.path.isdir(destPath):
                os.mkdir(destPath)
                print("falder made:",destPath)
                
            destPath = os.path.join(destPath, class_list[predict])
            if not os.path.isdir(destPath):
                os.mkdir(destPath)
                print("folder made:", destPath)
            img_name = test_x[i].split("\\")[-1]
            destFile = os.path.join(destPath, img_name)
            
            
            """
            shutil.copyfile(test_x[i], destFile)
            print("copy",test_x[i], "to", destFile)
            """
            result_list[i]['image'].save(destFile)
            
            
            destFile = os.path.join(destPath, img_name.replace("jpg", "txt"))
            with open(destFile, 'w', encoding='utf-8') as f:
                result = result_list[i]['result']
                for r in result:
                    r = [str(c) for c in r]
                    f.write(",".join(r))



