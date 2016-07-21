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

#include <stdio.h>
#include <time.h>

int main(int argc, char** argv) {
    cv::VideoCapture camera(0);

    upose::Context context(camera);

    time_t timer = time(0);
    unsigned int count;

    for(;;) {
        (void) context.step();

        ++count;

        printf("\rFPS: %f", count / difftime(time(0), timer));
        fflush(0);

        if(cv::waitKey(1) == 27) break;
    }
}
