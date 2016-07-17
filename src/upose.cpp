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
     * The extra division in there helps account for illumination.
     */

    cv::Mat Context::backgroundSubtract(cv::Mat frame) {
        cv::Mat foreground = cv::abs(m_background - frame) / frame;

        cv::cvtColor(foreground, foreground, CV_BGR2GRAY);
        cv::threshold(foreground, foreground, 0.1, 255, cv::THRESH_BINARY);

        return foreground;
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat foreground = backgroundSubtract(frame);
        cv::cvtColor(foreground, foreground, CV_GRAY2BGR);
        
        cv::imshow("You!", foreground & frame);
    }

    void Skeleton::visualize(cv::Mat image) {
        cv::circle(image, head2d(), 25, cv::Scalar(255, 0, 0), -1);
        cv::circle(image, neck2d(), 25, cv::Scalar(255, 255, 255), 5);
        cv::circle(image, lshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
        cv::circle(image, rshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
    }

}
