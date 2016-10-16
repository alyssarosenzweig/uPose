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

    /* the foreground and skin models are probabilistic;
     * that way, we can never make a mistake :-)
     * TODO: research proper Gaussian models
     * at the moment, just map to a sigmoid curve
     */

    cv::Mat Context::backgroundSubtract(cv::Mat frame) {
        /*
        cv::Mat foreground = cv::abs(m_background - frame);
        cv::cvtColor(foreground > 0.25*frame, foreground, CV_BGR2GRAY);

        cv::blur(foreground > 0, foreground, cv::Size(5, 5));
        cv::blur(foreground > 254, foreground, cv::Size(7, 7));
        return foreground > 254;*/

        cv::Mat foreground = cv::abs(m_background - frame);
        cv::cvtColor(foreground, foreground, CV_BGR2GRAY);
        foreground.convertTo(foreground, CV_64F);
        
        cv::Mat diff = -(foreground - 40) * 0.1;
        cv::exp(diff, diff);
        return 1. / (1 + diff);
    }

     /**
      * convert frame to (Y)I(Q) space for skin color thresholding
      * the Y and Q components are not necessary, however.
      * algorithm from Brand and Mason 2000
      * "A comparative assessment of three approaches to pixel level human skin-detection"
      *
      * TODO: again, research better probabilistic models
      * for now use a bell curve with μ = 10
      */

    cv::Mat Context::skinRegions(cv::Mat frame) {
        cv::Mat bgr[3];
        cv::split(frame, bgr);

        cv::Mat map = (0.6*bgr[2]) - (0.3*bgr[1]) - (0.3*bgr[0]);
        map.convertTo(map, CV_64F);

        map -= 10;

        cv::Mat diff = -map.mul(map) * 0.02;
        cv::exp(diff, diff);
        cv::imshow("DIFF", diff);
        return diff;
    }

    cv::Mat leftHandmap(cv::Size size, cv::Mat foreground,
                                       cv::Mat skin,
                                       cv::Point centroid) {
        cv::Mat map = cv::Mat::zeros(size, CV_32F);

        for(int x = 0; x < size.width; ++x) {
            for(int y = 0; y < size.height; ++y) {
                /* calculate probability */
                double p = 1;

                /* update with foreground model */
                p *= foreground.at<double>(y, x);

                /* update with skin model */
                p *= skin.at<double>(y, x);

                /* update with centroid X model */
                double dx = (x - centroid.x) - (-200);
                p *= exp(-dx*dx/90000);

                /* update with centroid Y model */
                double dy = (double) (y - centroid.y) - (-200);
                p *= exp(-dy*dy/90000);

                map.at<float>(y, x) = p;
            }
        }

        return map;
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat foreground = backgroundSubtract(frame);
        cv::Mat skin = skinRegions(frame);

        cv::imshow("Foreground", foreground);
        cv::imshow("Skin", skin);

        /* approximate probabilistic distance transform */
        int pdtConstant = 127;

        cv::Mat pdt;
        cv::boxFilter(foreground, pdt, -1, cv::Size(pdtConstant, pdtConstant),
                                           cv::Point(-1, -1), false);
        pdt = (pdt - foreground) / (pdtConstant*pdtConstant - 1);
        pdt = pdt.mul(foreground);
        pdt *= 255;
        pdt.convertTo(pdt, CV_8U);
        applyColorMap(pdt, pdt, cv::COLORMAP_JET);
        cv::imshow("PDT", pdt);

        cv::Moments centroidM = cv::moments(foreground);
        cv::Point centroid = cv::Point(
                centroidM.m10 / centroidM.m00,
                centroidM.m01 / centroidM.m00
            );

        cv::Mat handmap = leftHandmap(frame.size(), foreground, skin, centroid) * 255;
        handmap.convertTo(handmap, CV_8U);
        applyColorMap(handmap, handmap, cv::COLORMAP_JET);
        cv::imshow("Color", handmap);

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
