/** 
 * upose.h
 * header file for uPose
 *
 * Copyright (C) 2016 Alyssa Rosenzweig
 * ALL RIGHTS RESERVED
 */

#include <opencv2/opencv.hpp>

#define countof(arr) (sizeof(arr) / sizeof(arr[0]))

namespace upose {
    struct Features2D {
        cv::Point face;
        cv::Point neck, leftShoulder, rightShoulder;
        cv::Point leftHand, rightHand;
        cv::Point leftFoot, rightFoot;
    };

    typedef int UpperBodySkeleton[2 * 2];

    enum Skeleton2D {
        JOINT_ELBOWL = 0,
        JOINT_ELBOWR = 2,
    };

    void visualizeUpperSkeleton(cv::Mat image, Features2D f, UpperBodySkeleton skel);

    class Human {
        public:
            Human(cv::Mat _foreground, cv::Mat _skinRegions, cv::Mat _edgeImage, Features2D _projected) :
                                        foreground(_foreground),
                                        skinRegions(_skinRegions),
                                        edgeImage(_edgeImage),
                                        projected(_projected) {}

            cv::Mat foreground, skinRegions, edgeImage;
            Features2D projected;
    };

    class Context {
        public:
            Context(cv::VideoCapture& camera);

            void step();

        private:
            cv::VideoCapture& m_camera;

            cv::Mat m_background, m_lastFrame;
            cv::Mat backgroundSubtract(cv::Mat frame);
            cv::Mat skinRegions(cv::Mat frame);
            cv::Mat edges(cv::Mat frame);

            Features2D m_last2D, m_lastu2D;
            void track2DFeatures(cv::Mat foreground, cv::Mat skin);

            UpperBodySkeleton m_skeleton;
    };
}
