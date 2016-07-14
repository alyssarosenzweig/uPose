/**
 * upose.cpp
 * the uPose pose estimation library
 *
 * Copyright (C) 2016 Alyssa Rosenzweig
 * ALL RIGHTS RESERVED
 */

#include <opencv2/opencv.hpp>

namespace upose {
    /**
     * segments a human from an image
     * returns a binary mask
     */

    cv::Mat segmentHuman(cv::Mat& human) {
        cv::Mat dogTemp;
        cv::GaussianBlur(human, dogTemp, cv::Size(15, 15), 0);
        cv::Mat edges = human - dogTemp;

        cv::cvtColor(edges, edges, CV_BGR2GRAY);
        edges.convertTo(edges, CV_32F);

        cv::log(edges, edges);

        return edges * 0.1;
    }
}
