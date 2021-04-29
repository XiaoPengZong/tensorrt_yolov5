# tensorrt_yolov5_v4.0

This project aims to produce tensorrt engine for yolov5, and calibrate the model for INT8.

## Env
* Ubuntu 18.04
* Tesla T4
* CUDA 11.0
* Driver 450.80.02
* tensorrt 7.0.0.11

## Run method
### 1. generate wts
```
// git clone src code according to `Different versions of yolov5` above
// download https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt
// copy gen_wts.py into ultralytics/yolov5
// ensure the file name is yolov5s.pt and yolov5s.wts in gen_wts.py
// go to ultralytics/yolov5
python gen_wts.py
// a file 'yolov5s.wts' will be generated.
```

### 2. modify parameters
You can change some key parameter in yolov5.cpp just like below.   
```cpp
#define USE_FP16 // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
bool save_txt = false;  // save detection result into txt files
bool save_img = true;  // whether save the image results
```

==**Notice**==
> 1. if you set **USE_INT8** model, you must creat calibration_dataset, and put your dataset image in it. At least about 500 images can generate calibtate table.
> 2. `save_txt` means you can save detect result of every image, so that you can calculate the mAP of the model with [mAP](https://github.com/Cartucho/mAP#create-the-predicted-objects-files)


### 3. generate engine

```
// put yolov5s.wts into ./weights
// update CLASS_NUM in yololayer.h if your model is trained on custom dataset
mkdir build
cd build
cmake ..
make
./yolov5 -s 
```
After this step, you can get tensorrt engine named `yolov5s4.0_batch1.engine` according your batch size.

### 4. test images
You can set test image folder for below command.
```
./yolov5 -d [image folder]  
```
It will generate test result in `./experiment/images` folder.


## TODO
- [ ] Support for yolov5-v4.0 m/l/x
- [ ] Support for mAP test
- [ ] Comparison for tensorrt acceleration effect
- [ ] Run in deepstream project


## Reference

* https://github.com/wang-xinyu/tensorrtx
* https://github.com/ultralytics/yolov5
* https://github.com/Cartucho/mAP#create-the-predicted-objects-files

## Contributor
* @ [宗孝鹏](https://github.com/XiaoPengZong).
* @ [张波]()
* @ [于忠杰]()
* @ [杨叶]()