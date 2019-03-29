本项目使用的是商汤科技的mmdetection，链接如下https://github.com/open-mmlab/mmdetection, 关键使用了Retinanet101，具体配置参考config文件夹下Retina_r101.py，采用自定义的数据集

一、安装

(1)环境配置
Linux (tested on Ubuntu 16.04 and CentOS 7.2)
Python 3.4+
PyTorch 1.0
Cython
mmcv >= 0.2.2

(2)安装步骤  #上传的模型为已经安装后的版本，应该不需要再次安装，为保险起见，附上安装过程

先编译
cd code
cd mmdetection
pip install cython  
./compile.sh

后安装
python(3) setup.py install  # 或者用句号结尾安装 "pip install ."


二、数据准备

(1)解压数据
运行code目录下zip.py文件，解压的数据存放于data/First_round_data/目录下

(2)数据增广
运行code目录下data_augmentation.py，生成增广图集和新的标注.pkl文件，位置于mmdetection/data/coco/annotations

(3)测试文件目录
运行code目录下pickle_file_creation.py，读取测试集图片名


三、模型训练
在code/mmdetection/目录下，运行 
./tools/dist_train.sh ./config/retinanet_r101_fpn_1x.py  4 --validate   #其中4表示gpu数量


四、测试模型
在code/mmdetection/目录下，运行 
python tools/test.py config/retinanet_r101_fpn_1x.py work_dirs/fifi/epoch_20.pth --gpus 4  --final1.pkl

运行code目录下json_to_json.py，submit/目录中将生成最终上传的json文件final.json

