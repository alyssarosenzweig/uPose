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
     * this is a very crude algorithm at the moment,
     * and will probably break.
     * foreground? = (background - foreground)^2 > 255 roughly
     * TODO: use something more robust
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

        /* visualize error function */

        if(humans.size() == 1) {
            Skeleton guess;

            /* initialize center to mean */
            int totalX = 0, totalY = 0;

            for(int i = 0; i < humans[0].size(); ++i) {
                totalX += humans[0][i].x;
                totalY += humans[0][i].y;
            }

            guess.center.x = totalX / humans[0].size();
            guess.center.y = totalY / humans[0].size();

            /*double alpha = 0.01;
            
            for(int iteration = 0; iteration < 50; ++iteration) {
                Skeleton gradient = gradient2d(humans[0], guess);

                guess.head -= alpha *  
            }*/

            cv::circle(foreground, guess.center, 20, cv::Scalar(255, 0, 0), -1);
        }

        cv::imshow("Foreground", foreground);

        return humans;
    }

    Skeleton Context::cost2d(
            std::vector<cv::Point> human,
            Skeleton guess
    ) {
        /* stub */
    }

    Skeleton Context::gradient2d(
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
