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

        cv::blur(binary, temp, cv::Size(7, 7));
        
        cv::Mat edges = temp - binary;
        cv::threshold(edges, edges, 8, 255, cv::THRESH_BINARY);
        
        return edges;
    }

    /* identify humans in the foreground by size of contours */

    std::vector<std::vector<cv::Point> > Context::identifyHumans(cv::Mat foreground) {
        cv::Mat edges = binaryEdges(foreground);

        std::vector<std::vector<cv::Point> > contours, humans;
        cv::findContours(edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        for(int i = 0; i < contours.size(); ++i) {
            if(cv::arcLength(contours[i], true) > 256) {
                humans.push_back(contours[i]);
            }
        }

        /* visualize error function */
        cv::imshow("Foreground", foreground);

        if(humans.size() == 1) {
            // compute B
            int board = 10000;
            for(int j = 0; j < humans[0].size(); ++j) {
                if(humans[0][j].y < board)
                    board = humans[0][j].y;
            }
            
            cv::Mat heat = cv::Mat::zeros(foreground.rows, foreground.cols, CV_8U);

            for(int y = 0; y < foreground.rows; ++y) {
                for(int x = 0; x < foreground.cols; ++x) {
                    int accumulator = 0;

                    for(int i = 0; i < humans[0].size(); ++i) {
                        int dx = (humans[0][i].x - x);
                        int dy = (humans[0][i].y - y);
                        int db = y - board;

                        accumulator += (dx*dx) + (dy*dy) + (db*db);
                    }

                    accumulator >>= 20;
                    if(accumulator > 255) accumulator = 255;

                    heat.at<char>(y, x) = accumulator;
                }
            }

            cv::imshow("Heat", heat);
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
}
