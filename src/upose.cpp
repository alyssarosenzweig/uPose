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
        m_previousLHand = cv::Point(-1, -1);
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
        foreground.convertTo(foreground, CV_32F);
        
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
        map.convertTo(map, CV_32F);
        map -= 15;

        cv::Mat diff = -map.mul(map) * 0.01;
        cv::exp(diff, diff);
        return diff;
    }

    cv::Mat generateDeltaMap(cv::Size size, cv::Point pt, 
                             int k, int lx, int ux, int ly, int uy) {
        cv::Mat map = cv::Mat::zeros(size, CV_32F);

        float muX = k*(ux + lx) / 2, muY = k*(uy + ly) / 2;
        float sdX = (k*ux - muX),    sdY = (k*uy - muY);

        for(int x = 0; x < size.width; ++x) {
            for(int y = 0; y < size.height; ++y) {
                float dx = (x - pt.x) - muX,
                      dy = (y - pt.y) - muY;

                float theta = -(dx*dx + dy*dy) / (2 * (sdX*sdX + sdY*sdY));
                map.at<float>(y, x) = theta; 
            }
        }

        cv::exp(map, map);
        return map;
    }

    cv::Mat leftHandmap(cv::Size size, cv::Mat foreground,
                                       cv::Mat skin,
                                       cv::Mat fpdt,
                                       cv::Mat spdt,
                                       cv::Point centroid,
                                       cv::Point previous) {
        cv::Mat map = cv::Mat::zeros(size, CV_32F);

        /* TODO: generate these constants from the user */
        cv::Mat centroidMap = generateDeltaMap(size, centroid, 50, -3, 2, -2, 2),
                motionMap   = generateDeltaMap(size, previous, 50, -1, 1, -1, 1);

        cv::imshow("Centroid", centroidMap);

        for(int x = 0; x < size.width; ++x) {
            for(int y = 0; y < size.height; ++y) {
                /* calculate probability */
                float p = 1;

                /* update with foreground model */
                p *= foreground.at<float>(y, x);

                /* update with distance model */
                //p *= spdt.at<float>(y, x);

                /* update with skin model */
                p *= skin.at<float>(y, x);

                /* update with centroid model */
                p *= centroidMap.at<float>(y, x);

                /* update with motion model */
                if(previous.x != -1 && previous.y != -1) {
                //    p *= motionMap.at<float>(y, x);
                }

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

        /* approximate probabilistic distance transform */
        cv::Mat fpdt;
        cv::blur(foreground, fpdt, cv::Size(127, 127));
        fpdt = fpdt.mul(foreground);

        cv::imshow("FPDT", fpdt);
        cv::Mat corners;
        cv::cornerHarris(foreground, corners, 2, 3, 0.01);
        cv::imshow("Corners", corners * 4096);

        cv::Mat spdt;
        cv::blur(skin, spdt, cv::Size(127, 127));
        spdt = spdt.mul(skin);

        cv::Mat visualPDT;
        visualPDT = spdt * 255;
        visualPDT.convertTo(visualPDT, CV_8U);
        applyColorMap(visualPDT, visualPDT, cv::COLORMAP_JET);
        cv::imshow("PDT", visualPDT);

        cv::Moments centroidM = cv::moments(foreground);
        cv::Point centroid = cv::Point(
                centroidM.m10 / centroidM.m00,
                centroidM.m01 / centroidM.m00
            );

        cv::Mat handmap = leftHandmap(frame.size(), foreground, skin, fpdt, spdt, centroid, m_previousLHand) * 255;

        double confidence;
        cv::Point lhand;
        cv::minMaxLoc(handmap, NULL, &confidence, NULL, &lhand);

        handmap.convertTo(handmap, CV_8U);
        applyColorMap(handmap, handmap, cv::COLORMAP_JET);
        cv::imshow("Color", handmap);

        if(confidence > 127) {
            cv::circle(frame, lhand, 15, cv::Scalar(0, 0, 255), -1);
            m_previousLHand = lhand;
        }

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
