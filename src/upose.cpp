/**
 * upose.cpp
 * the uPose pose estimation library
 *
 * Copyright (C) 2016 Alyssa Rosenzweig
 * ALL RIGHTS RESERVED
 */

#include <opencv2/opencv.hpp>
#include <upose.h>

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
        /* stub */

        cv::CascadeClassifier classifier("./static/haarcascade_upperbody.xml");

        std::vector<cv::Rect> objects;
        classifier.detectMultiScale(human, objects, 1.1, 1);

        for(int i = 0; i < objects.size(); ++i) {
            cv::rectangle(human, objects[i], cv::Scalar(255, 0, 0));
        }
        printf("Objects: %d\n", objects.size());

        return human;
    }

    Context::Context(cv::VideoCapture& camera) : m_camera(camera) {
        initializeStaticBackground();
    }

    /**
     * background subtraction logic
     * this is a very crude algorithm at the moment,
     * and will probably break.
     * foreground? = (background - foreground)^2 > 255 roughly
     * TODO: use something more robust
     */

    void Context::initializeStaticBackground() {
        m_camera.read(m_background);
    }

    cv::Mat Context::backgroundSubtract(cv::Mat frame) {
        cv::Mat foreground = m_background - frame;
        cv::multiply(foreground, foreground, foreground);

        cv::cvtColor(foreground, foreground, CV_BGR2GRAY);
        cv::threshold(foreground, foreground, 254, 255, cv::THRESH_BINARY);

        return foreground;
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat delta = backgroundSubtract(frame);

        cv::cvtColor(delta, delta, CV_GRAY2BGR);
        cv::bitwise_and(frame, delta, delta);

        cv::imshow("Frame", frame);
        cv::imshow("Delta", delta);
    }
}
