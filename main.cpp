// Author: Iegor Toporov
//
// NOTE:
// All parameters that the user can change are showed by a " // To change if needed " by the row's side.
//
//

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "pugixml.hpp"
#include "help.h"

int main() {

    // ======== LOADING OF NECESSARY FILES ========

    // Load image for parking spots detection and test on car detection
    std::string parkingSpotDetectionImageFolder = "ParkingLot_Dataset/sequence0/frames/2013-02-24_10_05_04"; // To change if needed
    std::string parkingSpotDetectionImagePath = parkingSpotDetectionImageFolder + ".jpg";
    cv::Mat parkingSpotDetectionImage = loadImage(parkingSpotDetectionImagePath);
    std::string testImageFolder = "ParkingLot_Dataset/sequence1/frames/2013-02-22_07_15_01"; // To change if needed
    std::string testImagePath = testImageFolder + ".png";
    cv::Mat testImg = loadImage(testImagePath);

    // Load xml for all ground truths
    std::string parkingSpotDetectionXml = replaceWordInString(parkingSpotDetectionImageFolder, "frames", "bounding_boxes") + ".xml";
    pugi::xml_document parkingSpotDetectionTruth = loadXml(parkingSpotDetectionXml);

    // Load ground truth masks for segmentation
    std::string testMaskPath = replaceWordInString(testImageFolder, "frames", "masks") + ".png";
    cv::Mat testMaskImage = loadImage(testMaskPath);
    Mat grayTestMaskImage;
    cv::cvtColor(testMaskImage, grayTestMaskImage, COLOR_BGR2GRAY);

    // ======== PARKING SPOTS DETECTION ========
    // Using function parkingSpotsDetect()

    double overlapThreshold = 0.06; // To chage if needed
    std::vector<cv::Rect> parkingSpotsDetected = parkingSpotsDetect(parkingSpotDetectionImage, overlapThreshold);

    cv::Mat imageDetectedParkingSpots = plotBBToImage(parkingSpotDetectionImage, parkingSpotsDetected);
    cv::Mat testImageDetectedParkingSpots = plotBBToImage(testImg, parkingSpotsDetected);
    cv::Mat imageTrueParkingSpots = plotBBFromXml(parkingSpotDetectionTruth, parkingSpotDetectionImage);
    cv::Mat stackedParkingSpots = stackImages(imageDetectedParkingSpots, imageTrueParkingSpots);

    // ======== CAR DETECTION ========
    // Using function carDetect() inside checkCarsInBB()
    
    double minCarArea = 400.0; // To change if needed
    std::vector<cv::Rect> detectedCars = carDetect(testImg);
    std::vector<cv::Rect> detectedCarsInsideBB = checkCarsInBB(testImg, parkingSpotsDetected, minCarArea);
    cv::Mat testImgDetectedCars = plotDetectedCarsToImage(testImageDetectedParkingSpots, detectedCarsInsideBB);
    std::vector<int> properParkingsCount = countProperParkings(testImg, detectedCars, detectedCarsInsideBB);

    // ======== CAR SEGMENTATION =======
    // Using segmentCars()

    cv::Mat detectedSegmentedImage = segmentCars(testImg, parkingSpotDetectionImage);
    cv::Mat trueSegmentedImage = colorizeImageWithMask(testImg, grayTestMaskImage);
    cv::Mat segmentedImageStack = stackImages(detectedSegmentedImage, trueSegmentedImage);

    // ======== MINIMAP CREATION ========
    // Using createMinimap()

    std::vector<Rect> trueParkingsBB = BBfromXml(parkingSpotDetectionTruth);
    cv::Mat miniMap = createMinimap(testImg, parkingSpotsDetected, detectedCarsInsideBB);
    cv::Mat testImgMiniMap = imageWithMinimap(testImgDetectedCars, miniMap);

    // ======== COMPUTE METRICS ========

    std::vector<Rect> maskedBB = extractBBMask(grayTestMaskImage);

    // MeanIoU and meanAP between BB for detected parking spots and ground truth spots
    float iouThreshold1 = 0.5f; // To change if needed
    float meanIoU1 = computeMeanIoU(trueParkingsBB, parkingSpotsDetected);
    float meanAP1 = computeMeanAP({trueParkingsBB}, {parkingSpotsDetected}, iouThreshold1);

    // MeanIoU and meanAP between detected cars and ground truth
    float iouThreshold2 = 0.5f; // To change if needed
    float meanIoU2 = computeMeanIoU(maskedBB, detectedCars);
    float meanAP2 = computeMeanAP({maskedBB}, {detectedCars}, iouThreshold2);

    std::string firstRecap = "MeanIoU = " + std::to_string(meanIoU1) + "\n " + "MeanAP = " + std::to_string(meanAP1);
    std::string secondRecap = "MeanIoU = " + std::to_string(meanIoU2) + "\n " + "MeanAP = " + std::to_string(meanAP2);

    // ======== PLOTS ========

    // First image: parking spots detection and meanIoU & meanAP
    cv::putText(stackedParkingSpots, firstRecap, cv::Point(10, testImg.rows - 50), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::imshow("Detected Spots (Up) - True spots (Down)", stackedParkingSpots);
    //cv::imwrite("2013-02-24_17_55_12.jpg", stackedParkingSpots); // Saving image (optional)

    // Second image: properly and improperly parked cars and meanIoU & meanAP
    std::string properImproperString = "Number of properly parked cars: " + std::to_string(properParkingsCount.at(0)) + "\n " + "Number of improperly parked cars: " + std::to_string(properParkingsCount.at(1));
    cv::putText(testImgMiniMap, properImproperString, cv::Point(10, testImg.rows - 80), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(testImgMiniMap, secondRecap, cv::Point(10, testImg.rows - 50), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::imshow("Classification of parking slots: occupied (red), empty (green)", testImgMiniMap);
    cv::imwrite("2013-02-22_07_15_01_detect.png", testImgMiniMap); // Saving image (optional)

    // Third image: segmentation of the cars
    cv::imshow("Detected segmented cars (up) - True segmented cars (down)", segmentedImageStack);
    //cv::imwrite("2013-04-12_15_00_09_segment.png", segmentedImageStack); // Saving image (optional)

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}