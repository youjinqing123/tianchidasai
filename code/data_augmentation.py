import cv2
import pickle
import os
import random
import json
import numpy as np
import copy

json_data = open('./train_no_poly.json').read()
data = json.loads(json_data)
images = data['images']
annotations  = data['annotations']
file_path = '../data/First_round_data/jinnan2_round1_train_20190305/restricted/'
#file_path = './data/coco/train2017_/'
new_file_path = './mmdetection/restricted_final/'
restricted_images = os.listdir(file_path)
output = []

for index in range(len(images)):
    task = {
        'filename':images[index]['file_name'],
        'id':images[index]['id'],
        'width':images[index]['width'],
        'height': images[index]['height'],
        'ann':{}
    }
    output.append(task)

iter = 0
pb = 0.5
for index in range(len(output)):
    bboxarr = np.array([])
    labelarr = np.array([])
    while iter < len(annotations) and annotations[iter]['image_id'] == output[index]['id']:
        bboxtmp = np.array([[(annotations[iter]['bbox'][0]), (annotations[iter]['bbox'][1]),(annotations[iter]['bbox'][0])+ (annotations[iter]['bbox'][2]),
                             (annotations[iter]['bbox'][1]) + (annotations[iter]['bbox'][3])]], dtype='int')
        labeltmp = np.array([annotations[iter]['category_id']], dtype='int')
        if len(bboxarr) == 0:
            bboxarr = bboxtmp
            labelarr = labeltmp
        else:
            bboxarr = np.append(bboxarr, bboxtmp, axis = 0)
            labelarr = np.append(labelarr, labeltmp, axis=0)

        output[index]['ann']['bboxes'] = bboxarr
        output[index]['ann']['labels'] = labelarr
        iter = iter + 1
    if len(output[index]['ann']) != 0:
        img = cv2.imread(file_path + output[index]['filename'])
        if img is not None:
            cv2.imwrite(new_file_path + output[index]['filename'], img)
            # flip horizontally
            if random.random() > pb:
                result = cv2.flip(img, 1)
                cv2.imwrite(new_file_path + str(index) + '_h_flip.jpg', result)
                tmp = copy.deepcopy(output[index])
                tmp['filename'] = str(index) + '_h_flip.jpg'
                for i in range(tmp['ann']['bboxes'].shape[0]):
                    arr = copy.deepcopy(tmp['ann']['bboxes'][i])
                    arr[0] = tmp['width'] - tmp['ann']['bboxes'][i][2]
                    arr[2] = tmp['width'] - tmp['ann']['bboxes'][i][0]
                    tmp['ann']['bboxes'][i] = arr
                # cv2.rectangle(result, (arr[0], arr[1]), (arr[2], arr[3]), (0, 0, 255), 4)
                # cv2.imwrite('001_new.jpg', result)
                output.append(tmp)
            # flip vertically
            if random.random() > pb:
                result = cv2.flip(img, 0)
                cv2.imwrite(new_file_path + str(index) + '_v_flip.jpg', result)
                tmp = copy.deepcopy(output[index])
                tmp['filename'] = str(index) + '_v_flip.jpg'
                for i in range(tmp['ann']['bboxes'].shape[0]):
                    arr = copy.deepcopy(tmp['ann']['bboxes'][i])
                    arr[1] = tmp['height'] - tmp['ann']['bboxes'][i][3]
                    arr[3] = tmp['height'] - tmp['ann']['bboxes'][i][1]
                    tmp['ann']['bboxes'][i] = arr
                    #result = cv2.rectangle(result, (arr[0], arr[1]), (arr[2], arr[3]), (0, 0, 255), 4)
                output.append(tmp)
            # flip vertically and horizontally
            if random.random() > 0.9:
                result = cv2.flip(img, -1)
                cv2.imwrite(new_file_path + str(index) + '_hv_flip.jpg', result)
                tmp = copy.deepcopy(output[index])
                tmp['filename'] = str(index) + '_hv_flip.jpg'
                for i in range(tmp['ann']['bboxes'].shape[0]):
                    arr = copy.deepcopy(tmp['ann']['bboxes'][i])
                    arr[0] = tmp['width'] - tmp['ann']['bboxes'][i][2]
                    arr[2] = tmp['width'] - tmp['ann']['bboxes'][i][0]
                    arr[1] = tmp['height'] - tmp['ann']['bboxes'][i][3]
                    arr[3] = tmp['height'] - tmp['ann']['bboxes'][i][1]
                    tmp['ann']['bboxes'][i] = arr
                    #result = cv2.rectangle(result, (arr[0], arr[1]), (arr[2], arr[3]), (0, 0, 255), 4)
                output.append(tmp)
            # rotate 90
            if random.random() > pb:
                result = np.rot90(img)
                cv2.imwrite(new_file_path + str(index) + '_rotate_90.jpg', result)
                tmp = copy.deepcopy(output[index])
                tmp['filename'] = str(index) + '_rotate_90.jpg'
                for i in range(tmp['ann']['bboxes'].shape[0]):
                    arr = copy.deepcopy(tmp['ann']['bboxes'][i])
                    arr[0] = tmp['ann']['bboxes'][i][1]
                    arr[1] = tmp['width'] - tmp['ann']['bboxes'][i][2]
                    arr[2] = tmp['ann']['bboxes'][i][3]
                    arr[3] = tmp['width'] - tmp['ann']['bboxes'][i][0]
                    tmp['ann']['bboxes'][i] = arr
                    #result = cv2.rectangle(result, (arr[0], arr[1]), (arr[2], arr[3]), (0, 0, 255), 4)
                #cv2.imwrite('004_new.jpg', result)
                val = copy.deepcopy(tmp['width'])
                tmp['width'] = copy.deepcopy(tmp['height'])
                tmp['height'] = copy.deepcopy(val)

                output.append(tmp)
            # rotate 270
            if random.random() > pb:
                result = np.rot90(img, 3)
                cv2.imwrite(new_file_path + str(index) + '_rotate_270.jpg', result)
                tmp = copy.deepcopy(output[index])
                tmp['filename'] = str(index) + '_rotate_270.jpg'
                for i in range(tmp['ann']['bboxes'].shape[0]):
                    arr = copy.deepcopy(tmp['ann']['bboxes'][i])
                    arr[0] = tmp['height'] - tmp['ann']['bboxes'][i][3]
                    arr[1] = tmp['ann']['bboxes'][i][0]
                    arr[2] = tmp['height'] - tmp['ann']['bboxes'][i][1]
                    arr[3] = tmp['ann']['bboxes'][i][2]
                    tmp['ann']['bboxes'][i] = arr
                    #result = cv2.rectangle(result, (arr[0], arr[1]), (arr[2], arr[3]), (0, 0, 255), 4)

                val = copy.deepcopy(tmp['width'])
                tmp['width'] = copy.deepcopy(tmp['height'])
                tmp['height'] = copy.deepcopy(val)
                output.append(tmp)

for index in range(len(output)):
    del output[index]['id']

for index in range(len(output)):
    if index < len(output) and len(output[index]['ann']) == 0:
        print(output[index])
        del output[index]
        index = index - 1


with open('mmdetection/data/coco/annotations/coco_input_all_augm_final.pkl', 'wb') as outfile:
    pickle.dump(output, outfile)

random.shuffle(output)

output_80 = []
for iter in range(int(len(output)*0.8)):
    output_80.append(output[iter])

with open('mmdetection/data/coco/annotations/coco_input_augm_80_final.pkl', 'wb') as outfile:
    pickle.dump(output_80, outfile)

output_20 = []
for iter in range(int(len(output)*0.8),len(output)):
    output_20.append(output[iter])
with open('mmdetection/data/coco/annotations/coco_input_augm_20_final.pkl', 'wb') as outfile:
    pickle.dump(output_20, outfile)









