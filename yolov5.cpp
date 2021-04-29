#include <iostream>
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"

#define USE_FP16 // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
bool save_txt = false;  // save detection result into txt files
bool save_img = true;  // whether save the image results


// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

static int get_width(int x, float gw, int divisor = 8) {
    //return math.ceil(x / divisor) * divisor
    if (int(x * gw) % divisor == 0) {
        return int(x * gw);
    }
    return (int(x * gw / divisor) + 1) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) {
        return 1;
    } else {
        return round(x * gd) > 1 ? round(x * gd) : 1;
    }
}

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    // Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());
	
    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    //yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(BATCH_SIZE, INPUT_W, INPUT_H, "../calibration_dataset/", ("../weights/int8calib_b" + std::to_string(BATCH_SIZE) + ".table").c_str(), INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;
    // Don't need the network any more
    network->destroy();
    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, float& gd, float& gw, std::string& wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {

    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

/*        
s:
gd = 0.33;
gw = 0.50;

m:
gd = 0.67;
gw = 0.75;

l:
gd = 1.0;
gw = 1.0;

x:
gd = 1.33;
gw = 1.25; 
*/

cv::Mat drawBBox(std::vector<Yolo::Detection>& res, cv::Mat& img, std::vector<std::string>& categories)
{
    for (size_t j = 0; j < res.size(); j++)
    {
        int class_idx = (int)res[j].class_id;
        cv::Rect r = get_rect(img, res[j].bbox);
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(img, categories[class_idx], cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);   
    }
    return img;
}


int main(int argc, char** argv) {
    std::string img_dir = "";
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    bool infer_video = false;

    cudaSetDevice(DEVICE);

    std::string wts_name = "../weights/yolov5s4.0.wts";
    std::string engine_name = "../weights/yolov5s4.0_batch" + std::to_string(BATCH_SIZE) + ".engine";
    float gd = 0.33, gw = 0.50;
    
    std::vector<std::string> categories = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic_light",
        "fire_hydrant", "stop_sign", "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove", "skateboard", "surfboard",
        "tennis_racket", "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch",
        "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell_phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
        "hair_drier", "toothbrush" };
    

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, gd, gw, wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 3 && std::string(argv[1]) == "-d") {
        img_dir = argv[2];
        std::ifstream file(engine_name, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -s  [serialize engine model to file]" << std::endl;
        std::cerr << "./yolov5 -d {test_dataset}  [deserialize engine file and run inference]" << std::endl;
        return -1;
    }
    
    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (img_dir.find("mp4") != std::string::npos)
    	 infer_video = true;         // video input
    if(!infer_video)
    {
        // image prepare
        std::vector<std::string> file_names;
        if (read_files_in_dir(argv[2], file_names) < 0) {
            std::cerr << "read_files_in_dir failed." << std::endl;
            return -1;
        }
        // batch infer
        int fcount = 0;
        int batch_nums = 0;
        size_t total_time = 0.0;
        for (int f = 0; f < (int)file_names.size(); f++) {
            fcount++;
            batch_nums++;
            if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
            for (int b = 0; b < fcount; b++) {
                
                cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
                if (img.empty()) {
                    std::cerr << "image is empty." << std::endl;
                }
                cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
                int i = 0;
                for (int row = 0; row < INPUT_H; ++row) {
                    uchar* uc_pixel = pr_img.data + row * pr_img.step;
                    for (int col = 0; col < INPUT_W; ++col) {
                        data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                        data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                        data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                        uc_pixel += 3;
                        ++i;
                    }
                }
            }

            // Run inference
            auto start = std::chrono::system_clock::now();
            doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
            auto end = std::chrono::system_clock::now();

            auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Inference time for each batch (Latency): " << batch_time << " ms" << std::endl;
            total_time += batch_time;

            std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
            // nms
            for (int b = 0; b < fcount; b++) {
                auto& res = batch_res[b];
                nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
            }

            for (int b = 0; b < fcount; b++) {
                auto& res = batch_res[b];
                std::cout << "Object number is: " << res.size() << std::endl;
                // save txt for mAP testing 
                if(save_txt)
                {
                    // open file
                    std::string::size_type idx = file_names[f - fcount + 1 + b].find('.');
                    std::string txt_file = file_names[f - fcount + 1 + b].substr(0, idx) + ".txt";
                    std::ofstream destFile("../experiment/result_txt/" + txt_file, std::ios::out);
                    if(!destFile) {
                        std::cout << "Open file error!" << std::endl;
                        return 0;
                    }

                    for(size_t j = 0; j < res.size(); j++)
                    {
                        std::string originalClass = categories[(int)res[j].class_id];
    	            	// std::string replaceClass = RefactoredClass(vehicle, bicycle, pedestrian, road_sign, originalClass);
                        // class + conf + center_x + center_y + w + h -> x1 + y1 + x2 + y2
                        // box area
                        cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
                        cv::Rect r = get_rect(img, res[j].bbox);

                        destFile << originalClass << " " 
                                 << res[j].conf << " "
                                 << r.tl().x<< " "
                                 << r.tl().y<< " " 
                                 << r.br().x<< " " 
                                 << r.br().y << std::endl;
                        // std::cout << "class : " << categories[(int)res[j].class_id] << " box location : " << res[j].bbox[0]<< " " <<
                        // res[j].bbox[1]<< " " << res[j].bbox[2]<< " " << res[j].bbox[3] << std::endl;
                    }
                    destFile.close();
                }

                // Save detected result images.
                if (save_img)
                {
                    cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
                    cv::Mat display_img = drawBBox(res, img, categories);
                    cv::imwrite("../experiment/images/" + file_names[f - fcount + 1 + b], display_img);
                }
            }
            fcount = 0;
        }
        std::cout << "Average inference time for each batch[" << std::to_string(BATCH_SIZE) << "] (Latency): " << total_time / batch_nums << "ms" << std::endl;
    }
    else
    {
        std::cout << "video stream input !" << std::endl;
        cv::VideoCapture video(img_dir);
		if(!video.isOpened())
		{
			std::cout << "failed to open video file!" << std::endl;
			return -1;
		}
		int frame_num = video.get(cv::CAP_PROP_FRAME_COUNT);
		std::cout << "total frame number is: " << frame_num << std::endl;
		float fps = 0.0;
		int frame_width = int(video.get(cv::CAP_PROP_FRAME_WIDTH));
		int frame_height = int(video.get(cv::CAP_PROP_FRAME_HEIGHT));
		fps = video.get(cv::CAP_PROP_FPS);
        // temp fps
        float curr_fps;

		// output decoding
		cv::VideoWriter writer("./result.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));
		// inference
        int fcount = 0;
		cv::Mat img;
		for(int f = 0; f < frame_num-1; f++)
		{
            auto start = std::chrono::system_clock::now();
            fcount++;
            std::cout << "=============" << std::endl;
            if (fcount < BATCH_SIZE && f + 1 != frame_num) continue;
            for (int b = 0; b < fcount; b++) 
            {
                video>>img;
                if (img.empty()) continue;
                cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
                // image normalization 
                for (int i = 0; i < INPUT_H * INPUT_W; i++) 
                {
                    data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
                    data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
                    data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
                }
            }

            // Run inference
            // auto start = std::chrono::system_clock::now();
            doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
            // auto end = std::chrono::system_clock::now();
            // std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

            std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
            // nms
            for (int b = 0; b < fcount; b++) 
            {
                auto& res = batch_res[b];
                nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
            }
            for (int b = 0; b < fcount; b++)
            {
                auto& res = batch_res[b];
                // inference results
    	        std::cout << "object number is: " << res.size() << std::endl;
                // write image to file 
    	        cv::Mat display_img = drawBBox(res, img, categories);
    	        writer << display_img;
            }
            auto end = std::chrono::system_clock::now();
            curr_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            if(fps==0.0)
                fps = curr_fps;
            else
                fps = fps*0.95 + curr_fps*0.05;
            std::cout << "fps : " << fps << std::endl;
            fcount = 0;
		}
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
