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

        for(unsigned int i = 0; i < contours.size(); ++i) {
            if(cv::arcLength(contours[i], true) > 256) {
                humans.push_back(contours[i]);
            }
        }

        return humans;
    }

    Skeleton Context::computeSkeleton2D(std::vector<cv::Point> human) {
        Skeleton guess;

        guess.head = cv::Point(200, 200);
        guess.neck = cv::Point(200, 250);
        guess.lshoulder = cv::Point(150, 250);
        guess.rshoulder = cv::Point(250, 250);

        /* calculate the second derivative of slope */
        std::vector<double> slopeDoublePrime;

        for(unsigned int s = 0; s < human.size() - 3; ++s) {
            double m0 = (human[s + 1].y - human[s].y)
                      / (human[s + 1].x - human[s].x);
            
            double m1 = (human[s + 2].y - human[s + 1].y)
                      / (human[s + 2].x - human[s + 1].x);

            double m2 = (human[s + 3].y - human[s + 2].y)
                      / (human[s + 3].x - human[s + 2].x);

            slopeDoublePrime.push_back(m2 - (2 * m1) - m0);
        }

        /* gradient descent */

        for(int iteration = 0; iteration < 50; ++iteration) {
            /* compute partial derivatives */

            double partialX = 0.0, partialY = 0.0;

            for(unsigned int pt = 0; pt < human.size() - 3; ++pt) {
                partialX += (guess.head.x - human[pt].x);
                partialY += (guess.head.y - human[pt].y);
            }

            /* adjust model per learning rate */

            guess.head.x -= 0.0001 * partialX;
            guess.head.y -= 0.0001 * partialY;

            printf("%d: (%d, %d)\n", iteration, guess.head.x, guess.head.y);
        }

        return guess;
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat foreground = backgroundSubtract(frame);
        std::vector<std::vector<cv::Point> > humans = identifyHumans(foreground);

        if(humans.size() == 1) {
            Skeleton skeleton = computeSkeleton2D(humans[0]);
            skeleton.visualize(frame);
        }

        cv::imshow("Frame", frame);
    }

    void Skeleton::visualize(cv::Mat image) {
        cv::circle(image, head2d(), 25, cv::Scalar(255, 0, 0), -1);
        cv::circle(image, neck2d(), 25, cv::Scalar(255, 255, 255), 5);
        cv::circle(image, lshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
        cv::circle(image, rshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
    }

}
