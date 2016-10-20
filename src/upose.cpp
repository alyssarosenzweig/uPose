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
                             float muX, float sdX, float muY, float sdY) {
        cv::Mat map = cv::Mat::zeros(size, CV_32F);

        for(int y = 0; y < size.height; ++y) {
            float* data = (float*) (map.data + y * map.step);

            for(int x = 0; x < size.width; ++x) {
                float dx = (x - pt.x) - muX,
                      dy = (y - pt.y) - muY;

                data[x] = -(dx*dx + dy*dy) / (2 * (sdX*sdX + sdY*sdY));
            }
        }

        cv::exp(map, map);
        return map;
    }

    cv::Mat generateDeltaMap(cv::Size size, cv::Point pt, 
                             int k, int lx, int ux, int ly, int uy) {
        float muX = k*(ux + lx) / 2, muY = k*(uy + ly) / 2;
        float sdX = (k*ux - muX),    sdY = (k*uy - muY);

        return generateDeltaMap(size, pt, muX, sdX, muY, sdY);
    }

    cv::Mat handMap(cv::Size size, int side,
                                   cv::Mat foreground,
                                   cv::Mat skin,
                                   cv::Point centroid,
                                   Features2D previous) {
        /* TODO: generate these constants from the user */
        cv::Mat centroidMap = generateDeltaMap(size, centroid, 100, side, side * 3, -3, -1),
                motionMap   = generateDeltaMap(size, previous.leftHand.pt, 50, -1, 1, -1, 1);

        return foreground.mul(skin).mul(centroidMap);
    }
 
    cv::Mat shoulderMap(cv::Size size, int side,
                                       cv::Mat foreground,
                                       cv::Point centroid,
                                       Features2D previous) {
        cv::Mat centroidMap = generateDeltaMap(size, centroid, side*100, 100, -100, 50),
                motionMap   = generateDeltaMap(size, previous.leftShoulder.pt, 50, -1, 1, -1, 1);

        return foreground.mul(centroidMap);
    }

    cv::Point momentCentroid(cv::Mat mat) {
        cv::Moments moment = cv::moments(mat);
        return cv::Point(moment.m10 / moment.m00, moment.m01 / moment.m00);
    }

    void trackPoint(cv::Mat heatmap, TrackedPoint* pt) {
        cv::minMaxLoc(heatmap, NULL, &(pt->confidence), NULL, (&pt->pt));
    }

    cv::Mat visualizeMap(cv::Mat map) {
        cv::Mat visualization = map * 255;

        visualization.convertTo(visualization, CV_8U);
        cv::applyColorMap(visualization, visualization, cv::COLORMAP_JET);

        return visualization;
   }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat foreground = backgroundSubtract(frame);
        cv::Mat skin = skinRegions(frame);
        cv::Point centroid = momentCentroid(foreground);

        cv::Size s = frame.size();

        trackPoint(handMap(s, -1, foreground, skin, centroid, m_last2D), &m_last2D.leftHand);
        trackPoint(handMap(s, +1, foreground, skin, centroid, m_last2D), &m_last2D.rightHand);
        cv::imshow("LS", visualizeMap(shoulderMap(s, -1, foreground, centroid, m_last2D)));
        trackPoint(shoulderMap(s, -1, foreground, centroid, m_last2D), &m_last2D.leftShoulder);
        trackPoint(shoulderMap(s, +1, foreground, centroid, m_last2D), &m_last2D.rightShoulder);

        visualizeUpperSkeleton(frame, m_last2D);
    }

    void visualizeUpperSkeleton(cv::Mat canvas, Features2D f) {
        cv::Scalar c(0, 200, 0); /* color */
        int t = 5; /* line thickness */

        cv::circle(canvas, f.leftHand.pt, 15, c, -1);
        cv::circle(canvas, f.leftShoulder.pt, 15, c, -1);

        cv::circle(canvas, f.rightHand.pt, 15, c, -1);
        cv::circle(canvas, f.rightShoulder.pt, 15, c, -1);

        cv::imshow("Visualization", canvas);
    }
}
