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
    Context::Context(cv::VideoCapture& camera) : m_camera(camera) {
        m_camera.read(m_background);
    }

    cv::Mat Context::backgroundSubtract(cv::Mat frame) {
        cv::Mat foreground = cv::abs(m_background - frame);
        cv::cvtColor(foreground > 0.25*frame, foreground, CV_BGR2GRAY);

        cv::blur(foreground > 0, foreground, cv::Size(5, 5));
        cv::blur(foreground > 254, foreground, cv::Size(7, 7));
        return foreground > 254;
    }

     /**
      * convert frame to (Y)I(Q) space for skin color thresholding
      * the Y and Q components are not necessary, however.
      * algorithm from Brand and Mason 2000
      * "A comparative assessment of three approaches to pixel level human skin-detection"
      */

    cv::Mat Context::skinRegions(cv::Mat frame, cv::Mat foreground) {
        cv::Mat bgr[3];
        cv::split(frame, bgr);

        cv::Mat map = (0.6*bgr[2]) - (0.3*bgr[1]) - (0.3*bgr[0]);
        cv::Mat skin = (map > 4) & (map < 20);

        cv::Mat tracked = foreground & skin;

        cv::blur(tracked, tracked, cv::Size(3, 3));
        cv::blur(tracked > 254, tracked, cv::Size(9, 9));

        return tracked > 0;
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat visualization = frame.clone();
        
        cv::Mat foreground = backgroundSubtract(frame);
        cv::Mat skin = skinRegions(frame, foreground);

        cv::imshow("Frame", frame);
    }

    void visualizeUpperSkeleton(cv::Mat out, Features2D f) {
        cv::Scalar c(0, 200, 0); /* color */
        int t = 5; /* line thickness */

        /*cv::line(out, f.leftHand, f.leftElbow, c, t);
        cv::line(out, f.leftElbow, f.leftShoulder, c, t);*/
        cv::line(out, f.leftShoulder, f.leftHand, c, t);
        cv::line(out, f.leftShoulder, f.neck, c, t);

        /*cv::line(out, f.rightHand, f.rightElbow, c, t);
        cv::line(out, f.rightElbow, f.rightShoulder, c, t);*/
        cv::line(out, f.rightShoulder, f.rightHand, c, t);
        cv::line(out, f.rightShoulder, f.neck, c, t);

        cv::line(out, f.neck, f.face, c, t);
    }
}
