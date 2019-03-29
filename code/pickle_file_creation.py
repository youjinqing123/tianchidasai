import numpy as np
import json
import os
from PIL import Image
import pickle

file_path = '../data/First_round_data/jinnan2_round1_test_b_20190326/'
test_images = os.listdir(file_path)
test_output = []
for iter in range(len(test_images)):
    img = Image.open(file_path+test_images[iter])
    task = {
        'filename':test_images[iter],
        'width':img.size[0],
        'height':img.size[1],
        'ann': {}
            }
    test_output.append(task)
with open('coco_test_final_1.pkl', 'wb') as outfile:
    pickle.dump(test_output, outfile)



