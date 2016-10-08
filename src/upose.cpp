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
        m_lastFrame = m_background;

        for(unsigned int i = 0; i < countof(m_skeleton); ++i) {
            m_skeleton[i] = 0;
        }
   }

    cv::Mat Context::backgroundSubtract(cv::Mat frame) {
        cv::Mat foreground = cv::abs(m_background - frame);
        cv::cvtColor(foreground > 0.25*frame, foreground, CV_BGR2GRAY);

        cv::blur(foreground > 0, foreground, cv::Size(5, 5));
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
        cv::Mat skin = map > 1 & map < 16;

        cv::Mat tracked = foreground & skin;

        cv::blur(tracked, tracked, cv::Size(3, 3));
        cv::blur(tracked > 254, tracked, cv::Size(9, 9));

        return tracked > 0;
    }

    
    /* the hand is the farthest point from the point closest to the shoulder */
    cv::Point sleeveNormalize(std::vector<cv::Point> contour, cv::Point shoulder) {
        int bestDist = 100000, bestIndex = 0;

        for(unsigned int i = 0; i < contour.size(); ++i) {
            int dist = cv::norm(shoulder - contour[i]);

            if(dist < bestDist) {
                bestDist = dist;
                bestIndex = i;
            }
        }

        return contour[(bestIndex + contour.size()/2) % contour.size()];
    }

    /**
     * tracks 2D features only, in 2D coordinates
     * that is, the face, the hands, and the feet
     */

    void Context::track2DFeatures(cv::Mat skin) {
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(skin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> boundings;
        std::vector<cv::Point> centroids;
        std::vector<std::vector<int> > costs;

        if(contours.size() >= 3) {
            for(unsigned int i = 0; i < contours.size(); ++i) {
                cv::Rect bounding = cv::boundingRect(contours[i]);

                cv::Point centroid = (bounding.tl() + bounding.br()) * 0.5;

                centroids.push_back(centroid);
                boundings.push_back(bounding);

                int w = bounding.width;

                std::vector<int> cost;
                cost.push_back(cv::norm(m_last2D.face - centroid) + centroid.y - w);
                cost.push_back(cv::norm(m_lastu2D.leftHand - centroid) + centroid.x - w);
                cost.push_back(cv::norm(m_lastu2D.rightHand - centroid) + (skin.cols - centroid.x) - w);

                costs.push_back(cost);
            }

            std::vector<int> minCost = {
                    (skin.rows*skin.rows + skin.cols*skin.cols) / 64,
                    (skin.rows*skin.rows + skin.cols*skin.cols) / 64,
                    (skin.rows*skin.rows + skin.cols*skin.cols) / 64
            };

            std::vector<int> indices = { -1, -1, -1 };

            /* minimize errors */

            for(unsigned int i = 0; i < contours.size(); ++i) {
                for(unsigned int p = 0; p < 3; ++p) {
                    if(costs[i][p] < minCost[p]) {
                        minCost[p] = costs[i][p];
                        indices[p] = i;
                    }
                }
            }

            if(indices[0] > -1) m_last2D.face      = centroids[indices[0]];
            if(indices[1] > -1) m_lastu2D.leftHand  = centroids[indices[1]];
            if(indices[2] > -1) m_lastu2D.rightHand = centroids[indices[2]];

            /* assign shoulder positions relative to face */

            if(indices[0] > -1) {
                cv::Rect face = boundings[indices[0]];
                cv::Point neck = cv::Point(face.x, face.y + 2*face.width);

                m_last2D.neck = neck;
                m_last2D.leftShoulder = neck + cv::Point(-face.width / 2, 0);
                m_last2D.rightShoulder = neck + cv::Point(3*face.width / 2, 0);
            }

            /* adjust for sleeves */
            if(indices[1] > -1) {
                m_last2D.leftHand = sleeveNormalize(
                        contours[indices[1]], 
                        m_last2D.leftShoulder
                    );
            }

            if(indices[2] > -1) {
                m_last2D.rightHand = sleeveNormalize(
                        contours[indices[2]], 
                        m_last2D.rightShoulder
                    );
            }
        }
    }

    cv::Mat Context::edges(cv::Mat frame) {
        cv::Mat edges;
        cv::blur(frame, edges, cv::Size(3, 3));
        cv::Canny(edges, edges, 32, 32 * 2, 3);
        return edges;
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat visualization = frame.clone();
        
        cv::Mat foreground = backgroundSubtract(frame);
        cv::Mat skin = skinRegions(frame, foreground);
        cv::Mat outline = edges(foreground) | edges(skin);

        cv::imshow("Outline", outline);
        cv::imshow("Skin", skin);

        track2DFeatures(skin);

        visualizeUpperSkeleton(visualization, m_last2D, m_skeleton);
        cv::imshow("visualization", visualization);
    }

    void visualizeUpperSkeleton(cv::Mat out, Features2D f, UpperBodySkeleton skel) {
        cv::Scalar c(0, 200, 0); /* color */
        int t = 5; /* line thickness */

        cv::line(out, f.leftHand, jointPoint2(skel, JOINT_ELBOWL), c, t);
        cv::line(out, jointPoint2(skel, JOINT_ELBOWL), f.leftShoulder, c, t);
        cv::line(out, f.leftShoulder, f.neck, c, t);

        cv::line(out, f.rightHand, jointPoint2(skel, JOINT_ELBOWR), c, t);
        cv::line(out, jointPoint2(skel, JOINT_ELBOWR), f.rightShoulder, c, t);
        cv::line(out, f.rightShoulder, f.neck, c, t);

        cv::line(out, f.neck, f.face, c, t);
    }
}
