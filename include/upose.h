/** upose.h
 * header file for uPose
 *
 * Copyright (C) 2016 Alyssa Rosenzweig
 * ALL RIGHTS RESERVED
 */

#include <opencv2/opencv.hpp>

namespace upose {
    cv::Mat segmentStaticHuman(cv::Mat& image);

    class Context {
        public:
            Context(cv::VideoCapture& camera);

            void step();
    };
}
