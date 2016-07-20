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
     G* the constructor initializes background subtraction, 2d tracking
     */

    Context::Context(cv::VideoCapture& camera) : m_camera(camera) {
        m_camera.read(m_background);

        m_last2D.face = cv::Point(m_background.cols / 2, 0);
        m_last2D.leftHand = cv::Point(0, m_background.rows / 2);
        m_last2D.rightHand = cv::Point(m_background.cols, m_background.rows / 2);
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
        return I;
    }

    /**
     * tracks 2D features only, in 2D coordinates
     * that is, the face, the hands, and the feet
     */

    void Context::track2DFeatures(cv::Mat foreground, cv::Mat skin) {
        cv::Mat tracked = foreground & skin;

        cv::blur(tracked, tracked, cv::Size(15, 15));
        cv::threshold(tracked, tracked, 100, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(tracked, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        
        cv::cvtColor(tracked, tracked, CV_GRAY2BGR);

        std::sort(contours.begin(),
                 contours.end(),
                 [](std::vector<cv::Point> l, std::vector<cv::Point> r) {
                    return cv::boundingRect(l).width > cv::boundingRect(r).width;
                 });

        std::vector<cv::Point> centroids;
        std::vector<std::vector<int> > costs;

        if(contours.size() >= 3) {
            for(unsigned int i = 0; i < 3; ++i) {
                cv::Rect bounding = cv::boundingRect(contours[i]);
                cv::Point centroid = (bounding.tl() + bounding.br()) * 0.5;

                centroids.push_back(centroid);

                cv::Point dface = m_last2D.face - centroid,
                          dleftHand = m_last2D.leftHand - centroid,
                          drightHand = m_last2D.rightHand - centroid;

                std::vector<int> cost;
                cost.push_back(dface.dot(dface));
                cost.push_back(dleftHand.dot(dleftHand));
                cost.push_back(drightHand.dot(drightHand));

                costs.push_back(cost);
            }

            std::vector<int> minFace = {
                    (skin.rows*skin.rows + skin.cols*skin.cols) / 16,
                    (skin.rows*skin.rows + skin.cols*skin.cols) / 16,
                    (skin.rows*skin.rows + skin.cols*skin.cols) / 16
            };

            std::vector<int> indices = { -1, -1, -1 };

            for(unsigned int i = 0; i < 3; ++i) {
                for(unsigned int p = 0; p < 3; ++p) {
                    if(costs[i][p] < minFace[p]) {
                        minFace[p] = costs[i][p];
                        indices[p] = i;
                    }
                }
            }

            if(indices[0] > -1) m_last2D.face      = centroids[indices[0]];
            if(indices[1] > -1) m_last2D.leftHand  = centroids[indices[1]];
            if(indices[2] > -1) m_last2D.rightHand = centroids[indices[2]];
        }

        cv::imshow("Skin", tracked);
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat foreground = backgroundSubtract(frame);
        cv::Mat skin = skinRegions(frame);

        track2DFeatures(foreground, skin);

        cv::circle(frame, m_last2D.face, 10, cv::Scalar(0, 255, 0), -1);
        cv::circle(frame, m_last2D.leftHand, 10, cv::Scalar(255, 0, 0), -1);
        cv::circle(frame, m_last2D.rightHand, 10, cv::Scalar(0, 0, 255), -1);

        cv::imshow("Frame", frame);
    }

    void Skeleton::visualize(cv::Mat image) {
        cv::circle(image, head2d(), 25, cv::Scalar(255, 0, 0), -1);
        cv::circle(image, neck2d(), 25, cv::Scalar(255, 255, 255), 5);
        cv::circle(image, lshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
        cv::circle(image, rshoulder2d(), 25, cv::Scalar(0, 0, 255), -1);
    }

}
