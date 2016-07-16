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

            cv::Vec2i neck, shoulderL, shoulderR;

            cv::Mat visualize(cv::Mat image);

            cv::Point head2d() { return head; }
            cv::Point neck2d() { return head + neck; }
            cv::Point shoulderL() { return head + neck + shoulderL; }
            cv::Point shoulderR() { return head + neck + shoulderR; }
    };

    class Context {
        public:
            Context(cv::VideoCapture& camera);

            void step();

        private:
            cv::VideoCapture& m_camera;

            cv::Mat m_background;
            cv::Mat backgroundSubtract(cv::Mat frame);

            cv::Mat binaryEdges(cv::Mat binary);

            std::vector<std::vector<cv::Point> > identifyHumans(cv::Mat foreground);
    };
}
