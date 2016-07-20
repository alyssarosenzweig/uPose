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
     G* the constructor initializes background subtraction, among other tasks
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

     /**
      * convert frame to (Y)I(Q) space for skin color thresholding
      * the Y and Q components are not necessary, however.
      * algorithm from Brand and Mason 2000
      * "A comparative assessment of three approaches to pixel level human skin-detection"
     */

    cv::Mat Context::skinRegions(cv::Mat frame) {
        cv::Mat bgr[3];
        cv::split(frame, bgr);

        cv::Mat I = (0.596*bgr[2]) - (0.274*bgr[1]) - (0.322*bgr[0]);
        cv::threshold(I, I, 2, 255, cv::THRESH_BINARY);

        cv::cvtColor(I, I, CV_GRAY2BGR);
        return I;
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat foreground = backgroundSubtract(frame);
        cv::Mat skin = skinRegions(frame) & foreground;

        cv::Mat visualization = skin;

        cv::Mat sgrey;
        cv::cvtColor(skin, sgrey, CV_BGR2GRAY);

        cv::blur(sgrey, sgrey, cv::Size(15, 15));
        cv::threshold(sgrey, sgrey, 100, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(sgrey, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        for(unsigned int i = 0; i < contours.size(); ++i) {
            cv::Rect bounding = cv::boundingRect(contours[i]);

            if(bounding.width > 32 && bounding.height > 32) {
                cv::drawContours(visualization, contours, i, cv::Scalar(0, 0, 255), 10);
                cv::rectangle(visualization, bounding, cv::Scalar(0, 255, 0), 10);

                cv::Point centroid = (bounding.tl() + bounding.br()) * 0.5;
                cv::circle(visualization, centroid, 16, cv::Scalar(255, 0, 0), -1);
            }
        }

        cv::imshow("Streamed", visualization);

    }

    void Skeleton::visualize(cv::Mat image) {
        cv::circle(image, head2d(), 25, cv::Scalar(255, 0, 0), -1);
        cv::circle(image, neck2d(), 25, cv::Scalar(255, 255, 255), 5);
        cv::circle(image, lshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
        cv::circle(image, rshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
    }

}
