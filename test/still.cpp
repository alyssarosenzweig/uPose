/**
 * still.c
 * Performs pose estimation on a single, prerecorded image.
 * This file is part of uPose.
 *
 * Copyright (C) 2016 Alyssa Rosenzweig
 * ALL RIGHTS RESERVED
 *
 * Usage:
 * $ ./test/still [image]
 *
 * `image` must be in a supported OpenCV file type.
 *
 * To terminate the application, strike any key.
 *
 * No advanced logic is in this file;
 * it is a simple wrapper around the core library.
 *
 * Depends on OpenCV (to read and display images) and uPose.
 */

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <upose.h>

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cout << "Usage ./test/still [image]" << std::endl;
        return 1;
    }

    std::string file(argv[1]);
    cv::Mat image = cv::imread(file, CV_LOAD_IMAGE_COLOR);
    cv::Mat mask = upose::segmentStaticHuman(image);
    cv::imshow("Mask", mask);

    cv::waitKey(0);

    return 0;
}
