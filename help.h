// Author: Iegor Toporov

#ifndef HELP_H
#define HELP_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "pugixml.hpp"

using namespace cv;
using namespace std;

// Function to load image
Mat loadImage(const string& imgPath);

// Function to load xml
pugi::xml_document loadXml(const string& xmlPath);

// Function to replace word in a string
string replaceWordInString(const string& path, const string& toReplace, const string& replacement);

// Function to draw Bounding boxes to image
Mat plotBBToImage(const Mat& img, const vector<Rect>& rects);

// Function to return BB from xml (identical to the one above)
vector<Rect> BBfromXml(const pugi::xml_document& xmlDoc);

// Function to stack 2 images
Mat stackImages(const Mat& img1, const Mat& img2);

// Function to draw BB from xml
Mat plotBBFromXml(const pugi::xml_document& xml, const Mat& img);

// Function to draw BB in red
Mat plotDetectedCarsToImage(const Mat& img, const vector<Rect>& rects);

// Create minimap of the parkimg lot in Mat (same size as input image)
Mat createMinimap(const Mat& referenceImage, const vector<Rect>& referenceBoxes, const vector<Rect>& detectedBoxes);

// Function to add an overlay of a minimap to an image
Mat imageWithMinimap(const Mat& img, const Mat& minimap);

// Function to compute slant angle
double computeSlantAngle(int x1, int y1, int x2, int y2);

// Function to compute area
int computeArea(const Rect& rect);

// Function to compute intersection
Rect computeIntersection(const Rect& rect1, const Rect& rect2);

// Function to detect parking spots
vector<Rect> parkingSpotsDetect(const Mat& img, double overlapThreshold);

// Function to detect cars and return bounding boxes
vector<Rect> carDetect(const Mat& img);

// Function to apply Non-Maximum Suppression
vector<Rect> applyNMS(const vector<Rect>& boxes, float threshold);

// Function to check if a parking space contains a car
vector<Rect> checkCarsInBB(const Mat& image, const vector<Rect>& boundingBoxes, double minCarArea);

// Function to convert a rotated rectangle (defined by center, size, and angle) to an axis-aligned bounding rectangle
Rect rotatedRectToBoundingRect(Point2f center, Size2f size, float angle);

// Function to merge rects with threshold
vector<Rect> mergeRectangles(const vector<Rect>& rectangles, double overlapThreshold);

// Function to segment cars using difference between 2 images
Mat segmentCars(const Mat& image1, const Mat& image2);

// Function for ground truth mask on image
Mat colorizeImageWithMask(Mat& image, const Mat& maskImage);

// Function to calculate Intersection over Union (IoU) between two bounding boxes
float computeIoU(const Rect& boxA, const Rect& boxB);

// Function that counts proper and improper parkings: vector.at(0) = proper, vector.at(2) = improper
vector<int> countProperParkings(const Mat& image, const vector<Rect>& detectedCars, const vector<Rect> detectedCarsInsideBB);

// Function to check if car is properly parked
bool isProperlyParked(const Rect& testBox, const vector<Rect>& referenceBoxes);

// Function that returns the BB of the ground truth mask
vector<Rect> extractBBMask(const Mat& mask);

// Function to compute the mean of Intersection over Union (IoU)
float computeMeanIoU(const std::vector<cv::Rect>& gtBoxes, const std::vector<cv::Rect>& predBoxes);

// Function to compute the mean Average Precision (mAP)
float computeMeanAP(const std::vector<std::vector<cv::Rect>>& allGtBoxes, const std::vector<std::vector<cv::Rect>>& allPredBoxes, float iouThreshold);

// Function to calculate Average Precision (AP) 
float computeAveragePrecision(const vector<cv::Rect>& gtBoxes, const vector<cv::Rect>& predBoxes, float iouThreshold);

#endif // HELP_H