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

        /* convert frame to (Y)I(Q) space for skin color thresholding
         * the Y and Q components are not necessary, however.
         * algorithm from Brand and Mason 2000
         * "A comparative assessment of three
         * approaches to pixel level human skin-detection"
         */

        cv::Mat bgr[3];
        cv::split(frame, bgr);

        cv::Mat I = (0.596*bgr[2]) - (0.274*bgr[1]) - (0.322*bgr[0]);
        cv::threshold(I, I, 8, 255, cv::THRESH_BINARY);
        cv::cvtColor(I, I, CV_GRAY2BGR);

        cv::Mat foreground = backgroundSubtract(frame);
        //cv::cvtColor(foreground, foreground, CV_BGR2GRAY);

        cv::imshow("Foreground", frame & (I & foreground));
        cv::imshow("Discrepency", I & ~foreground);

/*        cv::blur(foreground, foreground, cv::Size(21, 21));
        cv::threshold(foreground, foreground, 1, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(foreground, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<cv::Point> > hulls(contours.size());

        
        for(unsigned int i = 0; i < contours.size(); ++i) {
            cv::convexHull(contours[i], hulls[i], false);

            //cv::drawContours(frame, hulls, i, cv::Scalar(255, 0, 0), 10);
            //cv::drawContours(frame, contours, i, cv::Scalar(0, 0, 255), 10);

            for(unsigned int j = 0; j < contours[i].size() - 1; ++j) {
                double theta = atan2(
                        contours[i][j+1].y - contours[i][j].y,
                        contours[i][j+1].x - contours[i][j].x
                    );

                double d = sinf(theta);
                double c = d*d < 0.001 ? 255 : 0;

                cv::line(frame, contours[i][j+1], contours[i][j], cv::Scalar(c, c, c), 15);
                                
            }
        }*/

        cv::imshow("M", frame);
    }

    void Skeleton::visualize(cv::Mat image) {
        cv::circle(image, head2d(), 25, cv::Scalar(255, 0, 0), -1);
        cv::circle(image, neck2d(), 25, cv::Scalar(255, 255, 255), 5);
        cv::circle(image, lshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
        cv::circle(image, rshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
    }

}
