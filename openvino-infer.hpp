#include <iostream>
#include "openvino/openvino.hpp"
#include "opencv2/opencv.hpp"

class OpenvinoInfer 
{
public:
    OpenvinoInfer();
    void set_onnx_model(std::string model_path, std::string device);
    cv::Mat letterbox(const cv::Mat &img);
    void infer(cv::Mat &input);

    const float IMAGE_HEIGHT = 640.0;
    const float IMAGE_WIDTH = 640.0;

    ov::Core core;
    ov::preprocess::PrePostProcessor *ppp;
    ov::CompiledModel compiled_model;
    ov::Shape input_shape;
    ov::InferRequest infer_request;
    std::shared_ptr<ov::Model> model;
    ov::Tensor input_tensor1;

    std::vector<int> class_name = {1,2,3,4,5,6,7,8,9,10,11,12};
};