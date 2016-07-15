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
    Context::Context(cv::VideoCapture& camera) : m_camera(camera) {
        initializeStaticBackground();
    }

    /**
     * background subtraction logic
     * this is a very crude algorithm at the moment,
     * and will probably break.
     * foreground? = (background - foreground)^2 > 255 roughly
     * TODO: use something more robust
     */

    void Context::initializeStaticBackground() {
        m_camera.read(m_background);
    }

    cv::Mat Context::backgroundSubtract(cv::Mat frame) {
        cv::Mat foreground = m_background - frame;
        cv::multiply(foreground, foreground, foreground);

        cv::cvtColor(foreground, foreground, CV_BGR2GRAY);
        cv::threshold(foreground, foreground, 254, 255, cv::THRESH_BINARY);

        return foreground;
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat delta = backgroundSubtract(frame);
        cv::cvtColor(delta, delta, CV_GRAY2BGR);

        //cv::imshow("Frame", frame);
        cv::imshow("Delta", delta);
    }
}
