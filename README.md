# tensorrt_yolov5 :100:

This project aims to produce tensorrt engine for yolov5, and calibrate the model for INT8.


## Env
* Ubuntu 18.04
* Tesla T4
* CUDA 10.2
* Driver 450.80.02
* tensorrt 7.0.0.11

## Run method
### 1. generate wts
```
cd tensorrt_yolov5
git clone -b v4.0 https://github.com/ultralytics/yolov5.git
cd ./yolov5
cp ../gen_wts.py .
wget https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt
// or you can download it from url
python gen_wts.py
// a file 'yolov5s.wts' will be generated
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

**Notice**
> 1. if you set **USE_INT8** model, you must creat calibration_dataset, and put your dataset image in it. At least about 500 images can generate calibtate table.
> 2. `save_txt` means you can save detect result of every image, so that you can calculate the mAP of the model with [mAP](https://github.com/Cartucho/mAP#create-the-predicted-objects-files)


### 3. generate engine

```
// put yolov5s.wts into ./weights
mkdir weights
cp ./yolov5/yolov5s.wts ./weights

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

## Benchmark

|          | BatchSize | Latency,ms | Throughput (1000/latency*batchsize) | Latency Speedup (TRT latency/original latency) | Througnput Speedup (TRT throughput/original throughput) |
|----------|:---------:|:----------:|:-----------------------------------:|------------------------------------------------|:-------------------------------------------------------:|
|  Pytorch |     1     |     20     |                  50                 |                                                |                                                         |
|          |     8     |     17     |                 470                 |                                                |                                                         |
|          |     16    |     18     |                 888                 |                                                |                                                         |
|          |     32    |     19     |                 1684                |                                                |                                                         |
| TensorRT |     1     |     4.9    |                 204                 |                      0.245                     |                          4.08x                          |
|          |     8     |     4.1    |                 1951                |                      0.241                     |                          4.14x                          |
|          |     16    |     3.8    |                 4210                |                      0.211                     |                          4.73x                          |
|          |     32    |     2.2    |                14545                |                      0.115                     |                          8.63x                          |
## TODO
- [ ] Support for yolov5-v4.0 m/l/x
- [ ] Support for mAP test
- [ ] Comparison for tensorrt acceleration effect
- [ ] Run in deepstream project
- [ ] QAT will increase in the future


## Reference

* https://github.com/wang-xinyu/tensorrtx
* https://github.com/ultralytics/yolov5
* https://github.com/Cartucho/mAP#create-the-predicted-objects-files

## Contributor
* @ [宗孝鹏](https://github.com/XiaoPengZong)
* @ [张波](https://github.com/nanmi)
* @ [于忠杰]()
* @ [杨叶]()