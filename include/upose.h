/** upose.h
 * header file for uPose
 *
 * Copyright (C) 2016 Alyssa Rosenzweig
 * ALL RIGHTS RESERVED
 */

#include <opencv2/opencv.hpp>

namespace upose {
    class Skeleton {
        public:
            cv::Point head;

            cv::Point neck, lshoulder, rshoulder;

            void visualize(cv::Mat image);

            cv::Point head2d() { return head; }
            cv::Point neck2d() { return neck; }
            cv::Point lshoulder2d() { return lshoulder; }
            cv::Point rshoulder2d() { return rshoulder; }

            cv::Vec2i neckD() { return neck - head; }
            cv::Vec2i lshoulderD() { return lshoulder - neck; }
            cv::Vec2i rshoulderD() { return rshoulder - neck; }
    };

    class Features2D {
        public:
            cv::Point face, leftHand, rightHand, leftFoot, rightFoot;
    };

    class Context {
        public:
            Context(cv::VideoCapture& camera);

            void step();

        private:
            cv::VideoCapture& m_camera;

            cv::Mat m_background, m_bgEdges;
            cv::Mat backgroundSubtract(cv::Mat frame);

            cv::Mat skinRegions(cv::Mat frame);

            Features2D m_last2D;
            void track2DFeatures(cv::Mat foreground, cv::Mat skin);

            bool m_initiated;
    };
}
