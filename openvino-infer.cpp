#include "openvino-infer.hpp"

double sigmoid(double x) {
        if(x>0)
            return 1.0 / (1.0 + exp(-x));
        else
            return exp(x) / (1.0 + exp(x));
    }

OpenvinoInfer::OpenvinoInfer()
{
    std::cout << "OpenvinoInfer constructor" << std::endl;
}

void OpenvinoInfer::set_onnx_model(std::string model_path, std::string device)
{
    std::cout << "set_onnx_model" << std::endl;
    // -------- Step 1. Initialize OpenVINO Runtime Core -------

    // -------- Step 2. Read a model --------
    model = core.read_model(model_path);

    // -------- Step 3. Loading a model to the device --------
    compiled_model = core.compile_model(model, device);

    // -------- Step 4. Create an infer request --------
    infer_request = compiled_model.create_infer_request();
}

void OpenvinoInfer::infer(cv::Mat &input)
{
    ov::Core core = ov::Core();
    std::shared_ptr<ov::Model> model = core.read_model("/home/zyicome/zyb/NeuralNetwork/openvino_test_two/armor.onnx");
    if(model == nullptr)
    {
        std::cout << "model is nullptr" << std::endl;
        return ;
    }
    // Step . Inizialize Preprocessing for the model
    //ov::Shape input_shape;
    /*input_shape = {1, 3,static_cast<unsigned long>(IMAGE_HEIGHT), static_cast<unsigned long>(IMAGE_WIDTH)};
   ov::preprocess::PrePostProcessor *ppp;
        ppp = new ov::preprocess::PrePostProcessor(model);
        // Specify input image format
        ppp->input().tensor().set_element_type(ov::element::f32).set_layout("NCHW").set_color_format(ov::preprocess::ColorFormat::BGR); 
        //ppp->input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::BGR).scale({255., 255., 255.});

        //  Specify model's input layout
        ppp->input().model().set_layout("NCHW");
        // Specify output results format
        ppp->output().tensor().set_element_type(ov::element::f32);
        // Embed above steps in the graph
        model = ppp->build();*/
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    auto input_port = compiled_model.input();
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    std::cout << "infer" << std::endl;
    cv::Mat resized_img = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC3);

    cv::resize(input, resized_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));

    // 转换数据类型到CV_32FC3
    cv::Mat img_float;
    resized_img.convertTo(img_float, CV_32FC3);

    cv::Mat letterbox_img = letterbox(input);
    float x_scale = letterbox_img.cols / IMAGE_WIDTH;
    float y_scale = letterbox_img.rows / IMAGE_HEIGHT;
    cv::Mat blob;
    cv::dnn::blobFromImage(letterbox_img, blob, 1.0 / 255.0, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), cv::Scalar(), true, false);
    /*cv::imshow("blob", blob);
    cv::waitKey(0);*/
    std::cout << "input_port.get_element_type(): " << input_port.get_element_type() << std::endl;
    std::cout << "input_port.get_shape(): " << input_port.get_shape() << std::endl;

    uchar* input_data = (uchar *)blob.data; // 创建一个新的float数组

    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    infer_request.set_input_tensor(input_tensor);
    std::cout << "input_tensor: " << input_tensor.get_shape() << std::endl;
    infer_request.infer();
    auto output0 = infer_request.get_output_tensor(0);
    //auto output1 = infer_request.get_output_tensor(1);

    cv::Mat output_buffer(output0.get_shape()[1], output0.get_shape()[2], CV_32F, output0.data());
    //cv::Mat proto(32,25600, CV_32F, output1.data<float>());
    //transpose(output_buffer, output_buffer);
    float score_threshold = 0.65;
    float nms_threshold = 0.5;
    float conf_threshold = 0.5;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Mat> mask_confs;

    /*int rows = output_buffer.size[2];
    int dimensions = output_buffer.size[1];
    output_buffer = output_buffer.reshape(1, dimensions);
    cv::transpose(output_buffer, output_buffer);
    std::cout << "rows: " << rows << std::endl;
    std::cout << "dimensions: " << dimensions << std::endl;*/

    std::cout << "infer" << std::endl;
    for(int i = 1;i<output_buffer.cols;i++)
    {   
        /*std::cout << "output_buffer.rows: " << output_buffer.rows << std::endl;
        std::cout << "output_buffer.cols: " << output_buffer.cols << std::endl;
        std::cout << "output_buffer.col(i).size(): " << output_buffer.col(i).size() << std::endl;*/
        cv::Mat scores = output_buffer.col(i).rowRange(4,16);
        cv::transpose(scores, scores);
        //std::cout << "infer" << std::endl;
        //std::cout << "output_buffer.row(i).size(): " << output_buffer.row(i).size() << std::endl;
        cv::Mat boxes_mat = output_buffer.col(i).rowRange(0,4);
        cv::transpose(boxes_mat, boxes_mat);
        for(int j = 0;j<scores.cols;j++)
        {
            //scores.at<float>(j) = sigmoid(scores.at<float>(j));
        }
        cv::Point class_id_point;
        double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id_point);
        if(maxClassScore > score_threshold)
        {
            std::cout << "maxClassScore: " << maxClassScore << std::endl;
            std::cout << "scores: " << scores << std::endl;
            class_scores.push_back(maxClassScore);
            class_ids.push_back(class_id_point.x);
            float cx = boxes_mat.at<float>(0, 0);
            float cy = boxes_mat.at<float>(0, 1);
            float w = boxes_mat.at<float>(0, 2);
            float h = boxes_mat.at<float>(0, 3);
            int left = int((cx - w / 2) * x_scale);
            int top = int((cy - h / 2) * y_scale);
            int width = int(w * x_scale);
            int height = int(h * y_scale);
            boxes.push_back(cv::Rect(left, top, width, height));
            std::cout << "get a box" << std::endl;
            std::cout << "class_id_point.x: " << class_id_point.x << std::endl;
            std::cout << "cx: " << cx << std::endl;
            std::cout << "cy: " << cy << std::endl;
            std::cout << "w: " << w << std::endl;
            std::cout << "h: " << h << std::endl;
            std::cout << "x_scale: " << x_scale << std::endl;
            std::cout << "y_scale: " << y_scale << std::endl;
            std::cout << "left: " << left << std::endl;
            std::cout << "top: " << top << std::endl;
            std::cout << "width: " << width << std::endl;
            std::cout << "height: " << height << std::endl;
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, class_scores, score_threshold, nms_threshold, indices);
    for(int i = 0;i<indices.size();i++)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        int class_id = class_ids[idx];
        std::string name = std::to_string(class_name[class_id]);
        float score = class_scores[idx];
        cv::rectangle(input, box, cv::Scalar(0, 255, 0), 2);
        cv::putText(input, name, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::imshow("input", input);
        cv::waitKey(0);
    }
}

cv::Mat OpenvinoInfer::letterbox(const cv::Mat &img)
{
    /*int w = img.cols;
    int h = img.rows;
    int new_w = IMAGE_WIDTH;
    int new_h = IMAGE_HEIGHT;
    float scale = std::min(float(new_w) / w, float(new_h) / h);
    int width = int(w * scale);
    int height = int(h * scale);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height));
    cv::Mat canvas = cv::Mat::zeros(new_h, new_w, CV_8UC3);
    int x_offset = (new_w - width) / 2;
    int y_offset = (new_h - height) / 2;
    resized.copyTo(canvas(cv::Rect(x_offset, y_offset, width, height)));*/

    int col = img.cols;
    int row = img.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    img.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

int main()
{
    OpenvinoInfer openvino_infer;
    openvino_infer.set_onnx_model("/home/zyicome/zyb/NeuralNetwork/openvino_test_two/armor.onnx", "CPU");
    cv::Mat img = cv::imread("/home/zyicome/zyb/pictures/armors/images/8.jpg");
    cv::Mat img_resized = openvino_infer.letterbox(img);
    cv::imshow("img", img);
    cv::imshow("img_resized", img_resized);
    cv::waitKey(0);
    openvino_infer.infer(img);
    return 0;
}