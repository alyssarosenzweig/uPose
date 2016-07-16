/** upose.h
 * header file for uPose
 *
 * Copyright (C) 2016 Alyssa Rosenzweig
 * ALL RIGHTS RESERVED
 */

#include <opencv2/opencv.hpp>

namespace upose {
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

            Skeleton cost2d(std::vector<cv::Point> human, Skeleton guess);
            Skeleton gradient2d(std::vector<cv::Point> human, Skeleton guess);
    };

    class Skeleton {
        public:
            cv::Point head;
            cv::Point shoulderL, neck, shoulderR;
            cv::Point elbowL, elbowR;
            cv::Point handL, handR;
    }
}
