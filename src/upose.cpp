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
            printf("M: %d\n", humans[0].size());
            // compute B
            int board = 10000;
            for(int j = 0; j < humans[0].size(); ++j) {
                if(humans[0][j].y < board)
                    board = humans[0][j].y;
            }
            
            int m = humans[0].size();

            /* compute point weights */
            double total_weight = 0, total_weightX = 0, total_weightY = 0;

            for(int i = 0; i < m; ++i) {
                int w = 0;

                for(int j = 0; j < m; ++j) {
                    int dx = humans[0][i].x - humans[0][j].x;
                    int dy = humans[0][i].y - humans[0][j].y;

                    w += (dx*dx + dy*dy);
                }

                w >>= 16; // big numbers break floating point stuff >_>

                double weight = 1.0 / (double) w;

                total_weight += weight;
                total_weightX += weight * humans[0][i].x;
                total_weightY += weight * humans[0][i].y;
            }

            /* minimize error function with gradient descent */

            double headX = 0.0f, headY = 0.0f;
            double alpha = 0.1f;

            for(int iterations = 0; iterations < 50; ++iterations) {
                double partialX = (headX * total_weight) - total_weightX;
                double partialY = (headY * total_weight) - total_weightY;

                headX -= alpha * partialX;
                headY -= alpha * partialY;

//                printf("(%f, %f)\n", headX, headY);
           }

            cv::circle(foreground, cv::Point(headX, headY), 10, cv::Scalar(0, 255, 0), -1);
        }

        cv::imshow("Foreground", foreground);

        return humans;
    }

    std::vector<double> Context::cost2d(
            std::vector<cv::Point> human,
            Skeleton guess
    ) {
        /* stub */
    }

    std::vector<double> Context::gradient2d(
            std::vector<cv::Point> human,
            Skeleton guess
    ) {
        /* stub */
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat foreground = backgroundSubtract(frame);
        std::vector<std::vector<cv::Point> > humans = identifyHumans(foreground);

        cv::imshow("Frame", frame);
    }
}
