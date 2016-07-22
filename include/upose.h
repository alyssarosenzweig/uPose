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
        cv::Point face, leftHand, rightHand, leftFoot, rightFoot;
    };

    typedef int UpperBodySkeleton[3 * 7];

    enum Skeleton {
        JOINT_HEAD = 0,
        JOINT_SHOULDERL = 3,
        JOINT_SHOULDERR = 6,
        JOINT_ELBOWL = 9,
        JOINT_ELBOWR = 12,
        JOINT_HANDL = 15,
        JOINT_HANDR = 18
    };

    void visualizeUpperSkeleton(cv::Mat image, UpperBodySkeleton skel);

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
