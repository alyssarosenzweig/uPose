/**
 * upose.cpp
 * the uPose pose estimation library
 *
 * Copyright (C) 2016 Alyssa Rosenzweig
 * ALL RIGHTS RESERVED
 */

#include <opencv2/opencv.hpp>

namespace upose {
    /**
     * segments a human from a single image.
     *
     * the idea is to support still image detection
     * existing real-time systems, such as:
     * "Real-Time 3D Human Pose Estimation from Monocular View with
     * Applications to Event Detection and Video Gaming" by Ke et al,
     * rely on a trained background model.
     * uPose does not make this assumption, hence this function
     *
     * segmentStaticHuman:
     * takes an image of a scene assumed to have a human in it.
     * returns a binary mask encompassing humans in the image.
     *
     * in the future, face recognition might be used to see if we get this far?
     *
     * TODO: determine if my dog should be included in this mask :-)
     * (In seriousness, background subtraction has failed this way.)
     */

    cv::Mat segmentStaticHuman(cv::Mat& human) {
        cv::Mat dogTemp;
        cv::GaussianBlur(human, dogTemp, cv::Size(15, 15), 0);
        cv::Mat edges = human - dogTemp;

        cv::cvtColor(edges, edges, CV_BGR2GRAY);
        edges.convertTo(edges, CV_32F);

        cv::log(edges, edges);

        return edges * 0.1;
    }
}
