#include <iostream>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

int main(int argc, char **argv) 
{
    cv::Mat trainingData, trainingLabels;

    cv::String pos_img_dir = "/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/dataset/bigger_dataset_128_64/pos/.*jpg";
    cv::String neg_img_dir = "/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/dataset/bigger_dataset_128_64/neg/.*jpg";

    vector<String> fn;
    glob(pos_img_dir,fn);
    for(size_t i=0; i<fn.size(); ++i)
    {
        Mat im = imread(fn[i],0);
        im.convertTo(im,CV_32F);
        trainingData.push_back(im.reshape(1,1));
        trainingLabels.push_back(1);
    }
    glob(neg_img_dir,fn);
    for(size_t i=0; i<fn.size(); ++i)
    {
        Mat im = imread(fn[i],0);
        im.convertTo(im,CV_32F);
        trainingData.push_back(im.reshape(1,1));
        trainingLabels.push_back(0);
    }
    std::cout << trainingLabels.total() <<  std::endl;
    
    
    Ptr<ml::SVM> m_classifier = cv::ml::SVM::create();
    // Training parameters:
    m_classifier->setType(cv::ml::SVM::C_SVC);
    m_classifier->setKernel(cv::ml::SVM::POLY);
    m_classifier->setGamma(3);    
    m_classifier->setDegree(3);
    
    std::cout << "PreTrain check " << std::endl;
    
    m_classifier->trainAuto(ml::TrainData::create(trainingData, ml::ROW_SAMPLE, trainingLabels));
    
    std::cout << "PostTrain check " << std::endl;
                                                                     
    return 0;
}
