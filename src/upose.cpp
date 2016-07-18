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
        cv::cvtColor(foreground, foreground, CV_GRAY2BGR);

        return foreground;
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat foreground = backgroundSubtract(frame);

        /* segment skin */
        /* I really want to replace this for so many reasons
         * notably, it assumes the skin color of the user (light in my case)
         * TODO: find a better algorithm
         */

        cv::Mat channels[3];
        cv::split(frame, channels);
        cv::Mat skinSpace = cv::abs(channels[1] - channels[2]);

        cv::Mat lowerSkin, upperSkin;
        cv::threshold(skinSpace, lowerSkin, 4, 255, cv::THRESH_BINARY);
        cv::threshold(skinSpace, upperSkin, 14, 255, cv::THRESH_BINARY_INV);
        
        cv::Mat skin;
        cv::blur(lowerSkin & upperSkin, skin, cv::Size(5, 5));
        cv::threshold(skin, skin, 1, 255, cv::THRESH_BINARY);
        cv::cvtColor(skin, skin, CV_GRAY2BGR);

        cv::imshow("M", foreground & skin);
    }

    void Skeleton::visualize(cv::Mat image) {
        cv::circle(image, head2d(), 25, cv::Scalar(255, 0, 0), -1);
        cv::circle(image, neck2d(), 25, cv::Scalar(255, 255, 255), 5);
        cv::circle(image, lshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
        cv::circle(image, rshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
    }

}
