/**
 * webcam.c
 * Pose estimation from the webcam.
 * This file is part of uPose.
 *
 * Copyright (C) 2016 Alyssa Rosenzweig
 * ALL RIGHTS RESERVED
 *
 * Usage:
 * $ ./test/webcam
 */

#include <opencv2/opencv.hpp>
#include <upose.h>

int main(int argc, char** argv) {
    cv::VideoCapture camera(0);

    upose::Context context(camera);

    for(;;) {
        (void) context.step();

        if(cv::waitKey(30) == 27) break;
    }
}
