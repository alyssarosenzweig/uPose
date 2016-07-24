/** 
 * upose.h
 * header file for uPose
 *
 * Copyright (C) 2016 Alyssa Rosenzweig
 * ALL RIGHTS RESERVED
 */

#include <opencv2/opencv.hpp>

namespace upose {
    struct Features2D {
        cv::Point face, leftShoulder, rightShoulder;
        cv::Point leftHand, rightHand;
        cv::Point leftFoot, rightFoot;
    };

    typedef int UpperBodySkeleton[2 * 7];

    enum Skeleton2D {
        JOINT_SHOULDERL = 0,
        JOINT_SHOULDERR = 2,
        JOINT_ELBOWL = 4,
        JOINT_ELBOWR = 6,
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

            cv::Mat m_background;
            cv::Mat backgroundSubtract(cv::Mat frame);

            cv::Mat m_lastFrame;

            cv::Mat skinRegions(cv::Mat frame);

            cv::Mat edges(cv::Mat frame);

            Features2D m_last2D;
            void track2DFeatures(cv::Mat foreground, cv::Mat skin);

            UpperBodySkeleton m_skeleton;

            bool m_initiated;
    };
}
