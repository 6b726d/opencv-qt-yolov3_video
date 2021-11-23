#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <opencv2/bgsegm.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <fstream>
#include "geometry.h"
#include <opencv2/dnn.hpp>
#include <thread>
#include <array>
#include "timer.h"

//http://24.196.110.155:80/mjpg/video.mjpg
//http://153.156.230.207:8081/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000
//http://153.156.230.207:8084/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000
//http://61.197.202.11:80/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000

// General
int frame_delay = 50;                       // milliseconds
bool open_window = false;
bool play_video = true;
bool creating_area = false;
bool created_area = false;
std::vector<geometry::point2d> area;        // for save data (creating)
std::vector<geometry::point2d> used_area;   // for load data (created) (calc)
std::vector<cv::Point> cv_polygon;          // for draw (creating)
std::vector<cv::Point> cv_used_polygon;     // for draw (created)

// DNN
// Load Names
std::string class_file = "coco.names";
// Configuration and Weights Yolo Files
cv::String modelConfiguration = "yolov3-tiny.cfg";
cv::String modelWeights = "yolov3-tiny.weights";
// Load Network
cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);

// Object Detection
std::vector<std::string> class_names;
float conf_threshold = 0.5f;
float nms_threshold = 0.3f;
int ni_width = 320;
int ni_height = 320;

// Rects
std::vector<std::vector<cv::Point>> contours;   // Tracking
std::vector<int> class_ids;                     // Detection
std::vector<float> confidences;                 // Detection
std::vector<cv::Rect> boxes;                    // Detection
std::vector<int> indices;                       // Detection


void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
     if  ( event == cv::EVENT_LBUTTONDOWN ) {
          std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
          area.push_back(geometry::point2d(x, y));
          cv_polygon.push_back(cv::Point(x, y));
     }
}

std::vector<std::string> separate_values(std::string input, char label) {
    std::stringstream inputstream(input);
    std::string word;
    std::vector<std::string> words;
    while (getline(inputstream, word, label)) {
        words.push_back(word);
    }
    return words;
}

bool import_area(std::string file_name) {
    bool completed = false;
    used_area.clear();
    cv_used_polygon.clear();
    std::vector<std::string> vline;
    std::string line;
    std::ifstream myfile(file_name);
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            vline = separate_values(line, ' ');
            used_area.push_back(geometry::point2d(stof(vline[0]), stof(vline[1])));
            cv_used_polygon.push_back(cv::Point(stoi(vline[0]), stoi(vline[1])));
        }
        myfile.close();
        std::cout << "The lists was generated correctly" << std::endl;
        completed = true;
    }
    else {
        std::cout << "Unable to open file" << std::endl;
        completed = false;
    }
    return completed;
}

bool export_area(std::string file_name) {
    bool completed = false;
    int area_size = area.size();
    std::ofstream myfile(file_name);
    if (myfile.is_open()) {
        //myfile << std::setprecision(7); //7 for float, 15 for double
        for (int i = 0; i < area_size; i++) {
            myfile << area[i].x << " " << area[i].y << "\n";
        }
        myfile.close();
        std::cout << "the lists were generated correctly" << std::endl;
        completed = true;
    }
    else {
        std::cout << "unable to open file" << std::endl;
        completed = false;
    }
    area.clear();
    cv_polygon.clear();
    return completed;
}

void process_frame_detection(cv::Mat object_detection, cv::dnn::Net net) {
    class_ids.clear();
    confidences.clear();
    boxes.clear();
    indices.clear();

    cv::Mat blob;
    cv::dnn::blobFromImage(object_detection, blob, 1 / 255.0, cv::Size(ni_width, ni_height), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    std::vector<cv::String> layerNames = net.getLayerNames();
    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::vector<cv::String> outLayerNames(outLayers.size());
    for (int i = 0; i < outLayers.size(); i++) {
        outLayerNames[i] = layerNames[outLayers[i] - 1];
    }
    net.forward(outs, outLayerNames);


    for (int i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point class_id_point;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
            if (confidence > conf_threshold) {
                int center_x = (int)(data[0] * object_detection.cols);
                int center_y = (int)(data[1] * object_detection.rows);
                int width = (int)(data[2] * object_detection.cols);
                int height = (int)(data[3] * object_detection.rows);
                int left = center_x - width / 2;
                int top = center_y - height / 2;
                class_ids.push_back(class_id_point.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));

            }
        }
    }
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
}

void draw_detection(cv::Mat object_detection) {
    for (int i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        cv::rectangle(object_detection, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), cv::Scalar(255, 0, 00), 3);
        cv::putText(object_detection, class_names[class_ids[idx]], cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
    }
}

void process_frame_motion_bg(cv::Mat video_frame, cv::Mat motion_detection, cv::Ptr<cv::bgsegm::BackgroundSubtractorGSOC> background_subtr_method) {
    cv::Mat fgMask, background, fgMask_process;

    video_frame.copyTo(motion_detection);

    // pass the frame to the background subtractor
    background_subtr_method->apply(video_frame, fgMask);
    // obtain the background without foreground mask
    background_subtr_method->getBackgroundImage(background);

    cv::Mat kernel = cv::Mat::ones(5, 5, CV_32F);
    morphologyEx(fgMask, fgMask_process, cv::MORPH_OPEN, kernel);
    morphologyEx(fgMask_process, fgMask_process, cv::MORPH_DILATE, kernel);

    //morphologyEx(fgMask_process, fgMask_process, cv::MORPH_OPEN, kernel);
    //morphologyEx(fgMask_process, fgMask_process, cv::MORPH_DILATE, kernel);

    cv::findContours(fgMask_process, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
}


void process_frame_motion(cv::Mat video_frame, cv::Mat motion_detection) {
    cv::Mat diff, gray, blur, thresh, dilated;
    cv::absdiff(video_frame, motion_detection, diff);
    cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5,5), 0);
    cv::threshold(blur, thresh, 20, 255, cv::THRESH_BINARY);
    cv::dilate(thresh, dilated, cv::Mat(), cv::Point(-1,-1), 2);
    cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
}


void draw_motion(cv::Mat motion_detection) {
    if (created_area) {
        cv::polylines(motion_detection, cv_used_polygon, true, cv::Scalar(0, 255, 0), 2);
    }

    for (int i = 0; i < contours.size(); ++i) {
        cv::Rect rect = boundingRect(contours[i]); //
        geometry::point2d rect_center(rect.x + rect.width/2.0f, rect.y + rect.height/2.0f);
        if ((created_area == true) and (geometry::point2d_inside_polygon2d(used_area, rect_center) == false)) {
            cv::putText(motion_detection, "Movimiento Fuera del Area", cv::Point(10, 25), cv::QT_FONT_NORMAL, 1, cv::Scalar(0, 0, 255), 2);
        }
        cv::rectangle(motion_detection, rect, cv::Scalar(0, 0, 255), 4); //
    }
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->il_camera->addItem(".avi");
    ui->il_camera->addItem(".mp4");
    // Object Detection
    // Load Names
    std::ifstream file(class_file);
    std::string line;
    while (getline(file, line)) class_names.push_back(line);
    // OPENCV => CPU, OPENCL | CUDA => CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA); // DNN_TARGET_OPENCV
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); // DNN_TARGET_CPU or DNN_TARGET_OPENCL <=> CPU or GPU(Intel)
}


MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pb_find_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "Abrir Video", "", "Archivo de Video (*.mp4 *.avi)");
    ui->le_video->setText(file_name);
}

// This is the main code function
void MainWindow::on_pb_open_clicked()
{ 
    open_window = true;
    play_video = true;
    creating_area = false;
    created_area = false;

    contours.clear();
    class_ids.clear();
    confidences.clear();
    boxes.clear();
    indices.clear();

    bool video_is_open = false;
    bool video_active = false;
    bool camera_active = false;
    bool motion_active = false;
    bool object_active = false;

    // Radio Button
    if (ui->rb_video->isChecked()) {
        video_active = true;
    }
    else {
        camera_active = true;
    }

    // Check Box
    if (ui->cb_motion->isChecked()) {
        motion_active = true;
    }
    if (ui->cb_object->isChecked()) {
        object_active = true;
    }

    if (video_active) {
        cv::VideoCapture video;
        cv::Mat video_frame;
        cv::Mat motion_detection;
        cv::Mat object_detection;
        cv::String video_name = ui->le_video->text().toStdString();
        video.open(video_name);
        try {
            if (video.isOpened() == true) {
                video_is_open = true;
            }
            else {
                throw(505);
            }
        }  catch (...) {
            QMessageBox messageBox;
            messageBox.critical(0,"Error", "Error al abrir el archivo.");
            messageBox.setFixedSize(500,200);
        }
        if (video_is_open) {
            cv::String window_name = "Video";
            cv::String md_wname = "Deteccion de Movimiento";
            cv::String od_wname = "Deteccion de Objetos";
            cv::namedWindow(window_name, cv::WINDOW_NORMAL);

            cv::Ptr<cv::bgsegm::BackgroundSubtractorGSOC> background_subtr_method = cv::bgsegm::createBackgroundSubtractorGSOC();

            while (true) {
                if (!open_window) {
                    cv::destroyWindow(window_name);
                    if (motion_active) {
                        cv::destroyWindow(md_wname);
                    }
                    if (object_active) {
                        cv::destroyWindow(od_wname);
                    }
                    break;
                }
                if (play_video) {
                    video >> video_frame;
                    video >> motion_detection;
                    video >> object_detection;
                }
                if (video_frame.empty() or motion_detection.empty() or object_detection.empty()) {
                    cv::destroyWindow(window_name);
                    if (motion_active) {
                        cv::destroyWindow(md_wname);
                    }
                    if (object_active) {
                        cv::destroyWindow(od_wname);
                    }
                    break;
                }
                else {
                    cv::resize(video_frame, video_frame, cv::Size(640, 360));
                    if (creating_area) {
                        cv::setMouseCallback(window_name, CallBackFunc, NULL);
                        cv::polylines(video_frame, cv_polygon, true, cv::Scalar(0, 255, 0), 2);
                    }
                    else {
                        cv::setMouseCallback(window_name, NULL, NULL);
                    }
                    cv::imshow(window_name, video_frame);
                    if (motion_active) {
                        cv::resize(motion_detection, motion_detection, cv::Size(640, 360));
                        //process_frame_motion_bg(video_frame, motion_detection, background_subtr_method);
                        process_frame_motion(video_frame, motion_detection);
                        draw_motion(motion_detection);
                        cv::imshow(md_wname, motion_detection);
                    }
                    if (object_active) {
                        cv::resize(object_detection, object_detection, cv::Size(640, 360));
                        process_frame_detection(object_detection, net);
                        draw_detection(object_detection);
                        cv::imshow(od_wname, object_detection);
                    }
                }
                if (cv::waitKey(frame_delay) == 27) {
                    cv::destroyWindow(window_name);
                    if (motion_active) {
                        cv::destroyWindow(md_wname);
                    }
                    if (object_active) {
                        cv::destroyWindow(od_wname);
                    }
                    break;
                }
            }
        }
    }
    if (camera_active) {
        bool writer_active = false;
        bool writer_open = false;
        cv::String video_name = ui->le_stream->text().toStdString();
        if (video_name.empty() == false) {
            video_name += ui->il_camera->currentText().toStdString();
            writer_active = true;
        }
        cv::VideoCapture stream;
        cv::Mat stream_frame;
        cv::VideoWriter writer;
        int cam_idx = ui->sp_cam_idx->value();
        stream.open(cam_idx);
        if (writer_active) {
            int frame_width = stream.get(cv::CAP_PROP_FRAME_WIDTH);
            int frame_height = stream.get(cv::CAP_PROP_FRAME_HEIGHT);
            int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            double fps = 30.0;
            writer.open(video_name, codec, fps, cv::Size(frame_width, frame_height), true);
            if (writer.isOpened()) {
                writer_open = true;
            }
        }

        try {
            if (stream.isOpened() == true) {
                video_is_open = true;
            }
            else {
                throw(505);
            }
        }  catch (...) {
            QMessageBox messageBox;
            messageBox.critical(0,"Error", "Error al iniciar la cámara.");
            messageBox.setFixedSize(500,200);
        }
        if (video_is_open) {
            cv::String window_name = "Camara";

            while (true) {
                if (!open_window) {
                    cv::destroyWindow(window_name);
                    break;
                }
                stream >> stream_frame;
                if (writer_open) {
                    writer.write(stream_frame);
                }
                cv::imshow(window_name, stream_frame);
                if (cv::waitKey(frame_delay) == 27) {
                    cv::destroyWindow(window_name);
                    break;
                }
            }
        }
    }
}

void MainWindow::on_pb_play_clicked()
{
    play_video = !play_video;
}


void MainWindow::on_pb_close_clicked()
{
    area.clear();
    used_area.clear();
    cv_polygon.clear();
    cv_used_polygon.clear();
    contours.clear();
    class_ids.clear();
    confidences.clear();
    boxes.clear();
    indices.clear();
    open_window = false;
}


void MainWindow::on_pb_quit_clicked()
{
    area.clear();
    used_area.clear();
    cv_polygon.clear();
    cv_used_polygon.clear();
    contours.clear();
    class_ids.clear();
    confidences.clear();
    boxes.clear();
    indices.clear();
    open_window = false;
    close();
}


void MainWindow::on_pb_f_area_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "Abrir Área", "", "Archivo de Texto (*.txt)");
    ui->le_f_area->setText(file_name);
}


void MainWindow::on_pb_l_area_clicked()
{
    std::string file_name = ui->le_f_area->text().toStdString();
    bool completed = import_area(file_name);
    if (completed) {
        if (used_area.size() > 2) {
            used_area.push_back(used_area[0]);
            created_area = true;
            QMessageBox messageBox;
            messageBox.information(0, "Información", "Archivo importado correctamente.");
            messageBox.setFixedSize(500,200);
        }
        else {
            QMessageBox messageBox;
            messageBox.critical(0,"Error", "Los puntos son menos de tres.");
            messageBox.setFixedSize(500,200);
        }
    }
    else {
        QMessageBox messageBox;
        messageBox.critical(0,"Error", "Error al abrir el archivo.");
        messageBox.setFixedSize(500,200);
    }
}


void MainWindow::on_pb_c_area_clicked()
{
    std::string file_name = ui->le_c_area->text().toStdString();
    if (creating_area) {
        creating_area = false;
        if (area.size() > 2) {
            if (file_name.empty() == false) {
                file_name += ".txt";
            }
            bool completed = export_area(file_name);
            if (completed) {
                QMessageBox messageBox;
                messageBox.information(0, "Información", "Archivo exportado correctamente.");
                messageBox.setFixedSize(500,200);
            }
            else {
                QMessageBox messageBox;
                messageBox.critical(0,"Error", "Error al guardar el archivo.");
                messageBox.setFixedSize(500,200);
            }
        }
        else {
            area.clear();
            QMessageBox messageBox;
            messageBox.critical(0,"Error", "Los puntos son menos de tres.");
            messageBox.setFixedSize(500,200);
        }
    }
    else {
        creating_area = true;
    }
}
