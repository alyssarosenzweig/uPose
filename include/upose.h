/** 
 * upose.h
 * header file for uPose
 *
 * Copyright (C) 2016 Alyssa Rosenzweig
 * ALL RIGHTS RESERVED
 */

#include <opencv2/opencv.hpp>

namespace upose {
    typedef struct {
        cv::Point pt;
        double confidence;
    } TrackedPoint;

    struct Features2D {
        TrackedPoint head;
        TrackedPoint neck, leftShoulder, rightShoulder;
        TrackedPoint leftElbow, rightElbow;
        TrackedPoint leftHand, rightHand;
    };

    void visualizeUpperSkeleton(cv::Mat image, Features2D f);

    class Context {
        public:
            Context(cv::VideoCapture& camera);

            void step();

        private:
            cv::VideoCapture& m_camera;

            cv::Mat m_background;
            cv::Mat backgroundSubtract(cv::Mat frame);
            cv::Mat skinRegions(cv::Mat frame);
            cv::Mat edges(cv::Mat frame);

            Features2D m_last2D;
    };
}
