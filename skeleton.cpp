/**
 * skeleton.cpp
 * Sandbox for developing a skeletonization algorithm
 * Copyright (C) 2016 Alyssa Rosenzweig
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc, char** argv) {
    Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    // find contours
    
    cv::imshow("Image", image);

    cv::waitKey(0);
}
