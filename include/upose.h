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

            std::vector<cv::Rect> identifyHumans(cv::Mat foreground);
    };
}
