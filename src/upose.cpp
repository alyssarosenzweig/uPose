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
        cv::cvtColor(foreground, foreground, CV_GRAY2BGR);

        std::vector<std::vector<cv::Point> > contours, humans;
        cv::findContours(edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        for(int i = 0; i < contours.size(); ++i) {
            if(cv::arcLength(contours[i], true) > 256) {
                humans.push_back(contours[i]);
            }
        }

        /* visualize error function */

        if(humans.size() == 1) {
            // compute B
            int board = 10000;
            for(int j = 0; j < humans[0].size(); ++j) {
                if(humans[0][j].y < board)
                    board = humans[0][j].y;
            }
            
            /* accumulate points */
            int m = humans[0].size();

            int accumulatorX = 0, accumulatorY = 0;
            for(int i = 0; i < m; ++i) {
                accumulatorX += humans[0][i].x;
                accumulatorY += humans[0][i].y;
            }

            /* minimize error function with gradient descent */

            float headX = 0.0f, headY = 0.0f;
            float alpha = 0.0001f;

            for(int iterations = 0; iterations < 50; ++iterations) {
                float partialX = (m * headX) - accumulatorX;
                float partialY = (m * headY) - accumulatorY - (m * (board - headY));

                headX -= alpha * partialX;
                headY -= alpha * partialY;
           }

            cv::circle(foreground, cv::Point(headX, headY), 10, cv::Scalar(0, 255, 0), -1);
        }

        cv::imshow("Foreground", foreground);

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
