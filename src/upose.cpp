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
        m_background.convertTo(m_background, CV_32FC3);
    }

    /**
     * background subtraction logic
     * |background - frame| / frame.
     * The extra division in there helps account for illumination.
     */

    cv::Mat Context::backgroundSubtract(cv::Mat frame) {
        frame.convertTo(frame, CV_32FC3);

        cv::Mat foreground;
        cv::absdiff(m_background, frame, foreground);
        cv::divide(foreground, frame, foreground);

        cv::cvtColor(foreground, foreground, CV_BGR2GRAY);
        cv::threshold(foreground, foreground, 0.5, 255, cv::THRESH_BINARY);
        foreground.convertTo(foreground, CV_8UC3);

        return foreground;
    }

    /* difference of blurs for edge detection on binary mask inputs */

    cv::Mat Context::binaryEdges(cv::Mat binary) {
        cv::Mat temp;

        cv::blur(binary, temp, cv::Size(7, 7));
        
        cv::Mat edges = temp - binary;
        cv::threshold(edges, edges, 8, 255, cv::THRESH_BINARY);
        
        return edges;
    }

    /* identify humans in the foreground by size of contours */

    std::vector<std::vector<cv::Point> > Context::identifyHumans(cv::Mat foreground) {
        cv::Mat edges = binaryEdges(foreground);
        cv::cvtColor(foreground, foreground, CV_GRAY2BGR);

        std::vector<std::vector<cv::Point> > contours, humans;
        cv::findContours(edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        for(int i = 0; i < contours.size(); ++i) {
            if(cv::arcLength(contours[i], true) > 256) {
                humans.push_back(contours[i]);
            }
        }

        return humans;
    }


    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat foreground = backgroundSubtract(frame);
        std::vector<std::vector<cv::Point> > humans = identifyHumans(foreground);

        cv::imshow("Frame", frame);
    }

    cv::Mat Skeleton::visualize(cv::Mat image) {
        cv::circle(image, head2d(), 25, cv::Scalar(255, 0, 0), -1);
        cv::circle(image, neck2d(), 25, cv::Scalar(255, 255, 255), 5);
        cv::circle(image, lshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
        cv::circle(image, rshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
    }

}
