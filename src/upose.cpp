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
     * implements a crude, random search to optimize a function
     * TODO: switch to an advanced optimization algorithm
     */

    void optimizeRandomSearch(
            int (*cost)(int*, void*), /* cost function to minimize */
            int dimension, /* dimension of cost function */
            int iterationCount, /* number of iterations to run */
            int radius, /* radius of hypersphere */
            int* optimum, /* on entry, initial configuration. on exit, minimum */
            void* context /* (read-only) data to be passed to the cost function */
        ) {
        size_t size = sizeof(int) * dimension;

        int* candidate = (int*) malloc(size);
        memcpy(candidate, optimum, size);

        int best = cost(optimum, context);

        for(int iteration = 0; iteration < iterationCount; ++iteration) {
            /* step algorithm */
            int dim = iteration % dimension;
            candidate[dim] = optimum[dim] + (rand() % radius);

            /* save if a better solution */
            int candidateCost = cost(candidate, context);

            if(candidateCost < best) {
                memcpy(optimum, candidate, size);
                best = candidateCost;
            } else {
                candidate[dim] = optimum[dim];
            }
        }

        free(candidate);
    }

    /**
     * Context class: maintains a skeletal tracking context
     * the constructor initializes background subtraction, 2d tracking
     */

    Context::Context(cv::VideoCapture& camera) : m_camera(camera) {
        m_camera.read(m_background);

        m_last2D.face = cv::Point(m_background.cols / 2, 0);
        m_last2D.leftHand = cv::Point(0, m_background.rows / 2);
        m_last2D.rightHand = cv::Point(m_background.cols, m_background.rows / 2);

        m_lastFrame = m_background;
   }

    /**
     * background subtraction logic
     * The extra division in there helps account for illumination.
     */

    cv::Mat Context::backgroundSubtract(cv::Mat frame) {
        cv::Mat foreground = cv::abs(m_background - frame);
        cv::cvtColor(foreground > 0.25*frame, foreground, CV_BGR2GRAY);

        return foreground > 0;
    }

     /**
      * convert frame to (Y)I(Q) space for skin color thresholding
      * the Y and Q components are not necessary, however.
      * algorithm from Brand and Mason 2000
      * "A comparative assessment of three approaches to pixel level human skin-detection"
      */

    cv::Mat Context::skinRegions(cv::Mat frame) {
        cv::Mat bgr[3];
        cv::split(frame, bgr);

        return (0.596*bgr[2]) - (0.274*bgr[1]) - (0.322*bgr[0]) > 2;
    }

    /**
     * tracks 2D features only, in 2D coordinates
     * that is, the face, the hands, and the feet
     */

    void Context::track2DFeatures(cv::Mat foreground, cv::Mat skin) {
        cv::Mat tracked = foreground & skin;

        cv::cvtColor(foreground, foreground, CV_GRAY2BGR);

        cv::blur(tracked, tracked, cv::Size(9, 9));
        tracked = tracked > 127;

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(tracked, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        
        std::sort(contours.begin(),
                 contours.end(),
                 [](std::vector<cv::Point> l, std::vector<cv::Point> r) {
                    return cv::boundingRect(l).width > cv::boundingRect(r).width;
                 });

        std::vector<cv::Point> centroids;
        std::vector<std::vector<int> > costs;

        if(contours.size() >= 3) {
            for(unsigned int i = 0; i < 3; ++i) {
                cv::Rect bounding = cv::boundingRect(contours[i]);
                cv::Point centroid = (bounding.tl() + bounding.br()) * 0.5;

                centroids.push_back(centroid);

                std::vector<int> cost;
                cost.push_back(cv::norm(m_last2D.face - centroid) + centroid.y);
                cost.push_back(cv::norm(m_last2D.leftHand - centroid) + centroid.x);
                cost.push_back(cv::norm(m_last2D.rightHand - centroid) + (foreground.cols - centroid.x));

                costs.push_back(cost);
            }

            std::vector<int> minFace = {
                    (skin.rows*skin.rows + skin.cols*skin.cols) / 64,
                    (skin.rows*skin.rows + skin.cols*skin.cols) / 64,
                    (skin.rows*skin.rows + skin.cols*skin.cols) / 64
            };

            std::vector<int> indices = { -1, -1, -1 };

            for(unsigned int i = 0; i < 3; ++i) {
                for(unsigned int p = 0; p < 3; ++p) {
                    if(costs[i][p] < minFace[p]) {
                        minFace[p] = costs[i][p];
                        indices[p] = i;
                    }
                }
            }

            if(indices[0] > -1) m_last2D.face      = centroids[indices[0]];
            if(indices[1] > -1) m_last2D.leftHand  = centroids[indices[1]];
            if(indices[2] > -1) m_last2D.rightHand = centroids[indices[2]];
        }
    }

    cv::Mat Context::edges(cv::Mat frame) {
        cv::Mat edges;
        cv::blur(frame, edges, cv::Size(3, 3));
        cv::Canny(edges, edges, 32, 32 * 2, 3);
        return edges;
    }

    int projectJoint(int* joints, int index, int Y) {
        int Z = joints[index + 2];

        if(Z == 0 || Z < 0) return 0;

        return joints[index + Y] / joints[index + 2];
    }

    cv::Point jointPoint(int* joints, int index) {
        return cv::Point(
                projectJoint(joints, index, 0),
                projectJoint(joints, index, 1)
            );
    }

    int costFunction3D(UpperBodySkeleton skel, void* humanPtr) {
        Human* human = (Human*) humanPtr;

        /* stub */

        int lhandCostX = (skel[JOINT_HANDL + 0] - human->projected.leftHand.x);
        int lhandCostY = (skel[JOINT_HANDL + 1] - human->projected.leftHand.y);

        return (lhandCostX * lhandCostX) + (lhandCostY * lhandCostY);
    }

    void Context::step() {
        cv::Mat frame;
        m_camera.read(frame);

        cv::Mat visualization = frame.clone();
        
        cv::Mat foreground = backgroundSubtract(frame);
        cv::Mat skin = skinRegions(frame);
        cv::Mat edgeImage = edges(frame) & foreground;
        cv::Mat motion = cv::abs(frame - m_lastFrame);
        cv::cvtColor(motion, motion, CV_BGR2GRAY);

        if(m_initiated) {
            track2DFeatures(foreground, skin);

            Human human(foreground, skin, m_last2D);

            UpperBodySkeleton optima;
            optimizeRandomSearch(costFunction3D, sizeof(optima) / sizeof(int), 800, 30, optima, (void*) &human);

            visualizeUpperSkeleton(visualization, optima);
        } else {
            m_initiated = foreground.at<char>(foreground.cols/2, foreground.rows/2);
        }

        cv::imshow("visualization", visualization);

        m_lastFrame = frame.clone();
    }

    void visualizeUpperSkeleton(cv::Mat image, UpperBodySkeleton skel) {
        cv::circle(image, jointPoint(skel, JOINT_HANDL), 25, cv::Scalar(0, 0, 255), -1);
    }
}
