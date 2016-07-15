/**
 * upose.cpp
 * the uPose pose estimation library
 *
 * Copyright (C) 2016 Alyssa Rosenzweig
 * ALL RIGHTS RESERVED
 */

#include <opencv2/opencv.hpp>
#include <upose.h>

namespace upose {
    /**
     * Context class: maintains a skeletal tracking context
     * the constructor initializes background subtraction, among other tasks
     */

    Context::Context(cv::VideoCapture& camera) : m_camera(camera) {
        m_camera.read(m_background);
    }

    /**
     * background subtraction logic
     * this is a very crude algorithm at the moment,
     * and will probably break.
     * foreground? = (background - foreground)^2 > 255 roughly
     * TODO: use something more robust
     */

    cv::Mat Context::backgroundSubtract(cv::Mat frame) {
        cv::Mat foreground = m_background - frame;
        cv::multiply(foreground, foreground, foreground);

        cv::cvtColor(foreground, foreground, CV_BGR2GRAY);
        cv::threshold(foreground, foreground, 254, 255, cv::THRESH_BINARY);

        return foreground;
    }

    /* difference of blurs for edge detection on binary mask inputs */

    cv::Mat Context::binaryEdges(cv::Mat binary) {
        cv::Mat temp;

        cv::blur(binary, temp, cv::Size(3, 3));
        
        cv::Mat edges = temp - binary;
        cv::threshold(edges, edges, 32, 255, cv::THRESH_BINARY);
        
        return edges;
    }

    /* identify humans in the foreground by size of contours */

    std::vector<cv::Rect> Context::identifyHumans(cv::Mat foreground) {
        std::vector<cv::Rect> humans;

        cv::imshow("Edges", binaryEdges(foreground));

        return humans;
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat foreground = backgroundSubtract(frame);
        std::vector<cv::Rect> humans = identifyHumans(foreground);

        cv::imshow("Foreground", foreground);
    }
}
