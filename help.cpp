// Author: Iegor Toporov
// 
// NOTE:
// All functions are organized based on functionality group. This is not true for the organization of help.h
//
//

#include "help.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>
#include "pugixml.hpp"


using namespace cv;
using namespace std;

// ===================================
// LOADING FUNCTIONS

// Function to load image in a Mat
Mat loadImage(const string& imgPath) {
    
    Mat imgMat = imread(imgPath, IMREAD_COLOR);
    if (imgMat.empty()) {  
        std::cerr << "Failed to load reference image file: " << imgPath << std::endl;
        Mat emptyImage = Mat::zeros(0, 0, CV_8UC3);
        return emptyImage;
    }
    return imgMat;

}

// Function to load an xml into xml_document
pugi::xml_document loadXml(const string& xmlPath) {

    pugi::xml_document doc;
    pugi::xml_document emptyDoc;
    if (!doc.load_file(xmlPath.c_str())) {
        std::cerr << "Failed to load reference XML file: " << xmlPath << std::endl;
        return emptyDoc;
    }
     return doc;

}
// ===================================




// ===================================
// FUNCTIONS FOR STRINGS

// Function that replaces a word in a string
string replaceWordInString(const string& path, const string& toReplace, const string& replacement) {

    string copyPath = path;
    size_t pos = copyPath.find(toReplace);
    if (pos != std::string::npos) {
        copyPath.replace(pos, toReplace.length(), replacement);
    }
    return copyPath;

}
// ===================================




// ===================================
// FUNCTIONS FOR PLOTTING

// Function to plot BB to image
Mat plotBBToImage(const Mat& img, const vector<Rect>& rects) {

    Mat copyImg = img.clone();
    for (const auto& box : rects) {
        rectangle(copyImg, box, cv::Scalar(0, 255, 0), 2); // Green BB
    }
    return copyImg;

}

// Function to stack 2 images
Mat stackImages(const Mat& img1, const Mat& img2) {

    if (img1.cols != img2.cols) {
        resize(img2, img2, Size(img1.cols, img2.rows));
    }
    Mat stackedImage;
    vconcat(img1, img2, stackedImage);
    return stackedImage;

}

// Function to draw BB from xml
Mat plotBBFromXml(const pugi::xml_document& xmlDoc, const Mat& img) {

    Mat copyImg = img.clone();
    std::vector<cv::Rect> boundingBoxes_reference;  // Reference bounding boxes
    boundingBoxes_reference.clear(); 
    for (pugi::xml_node space : xmlDoc.child("parking").children("space")) {
        float centerX = space.child("rotatedRect").child("center").attribute("x").as_float();
        float centerY = space.child("rotatedRect").child("center").attribute("y").as_float();
        float width = space.child("rotatedRect").child("size").attribute("w").as_float();
        float height = space.child("rotatedRect").child("size").attribute("h").as_float();
        float angle = space.child("rotatedRect").child("angle").attribute("d").as_float();

        Point2f center(centerX, centerY);
        Size2f size(width, height);
        
        // Convert reference bounding boxes to rectangular
        Rect boundingBox_reference = rotatedRectToBoundingRect(center, size, angle);  
        boundingBoxes_reference.push_back(boundingBox_reference);  // Save the bounding boxes

        // Draw the bounding box in green
        rectangle(copyImg, boundingBox_reference, Scalar(0, 255, 0), 2); // Green color for bounding boxes
    }
    return copyImg;
}

// Function to draw BB from xml
vector<Rect> BBfromXml(const pugi::xml_document& xmlDoc) {

    std::vector<cv::Rect> boundingBoxes_reference;  // Reference bounding boxes
    boundingBoxes_reference.clear(); 
    for (pugi::xml_node space : xmlDoc.child("parking").children("space")) {
        float centerX = space.child("rotatedRect").child("center").attribute("x").as_float();
        float centerY = space.child("rotatedRect").child("center").attribute("y").as_float();
        float width = space.child("rotatedRect").child("size").attribute("w").as_float();
        float height = space.child("rotatedRect").child("size").attribute("h").as_float();
        float angle = space.child("rotatedRect").child("angle").attribute("d").as_float();

        Point2f center(centerX, centerY);
        Size2f size(width, height);
        
        // Convert reference bounding boxes to rectangular
        Rect boundingBox_reference = rotatedRectToBoundingRect(center, size, angle);  
        boundingBoxes_reference.push_back(boundingBox_reference);  // Save the bounding boxes
    }
    return boundingBoxes_reference;
}

// Function to plot BB in Red
Mat plotDetectedCarsToImage(const Mat& img, const vector<Rect>& rects) {

    Mat copyImg = img.clone();
    for (const auto& box : rects) {
        rectangle(copyImg, box, cv::Scalar(0, 0, 255), 2); // Red BB
    }
    return copyImg;

}

// Function to create a minimap plot of the parking lot
Mat createMinimap(const Mat& referenceImage, const vector<Rect>& referenceBoxes, const vector<Rect>& detectedBoxes) {
    // Create a white background image with the same size as the reference image
    Mat minimapPlot(referenceImage.size(), CV_8UC3, Scalar(255, 255, 255)); // White background

    // Define colors
    Scalar carColor(0, 0, 255);    // Red for properly parked cars
    Scalar emptyColor(255, 0, 0);  // Blue for empty spaces

    // Define IoU threshold
    float iouThreshold = 0.2;

    // Track matched reference boxes
    vector<bool> matched(referenceBoxes.size(), false);

    Rect bb;

    // Initialize counters
    int emptySpaces = 0;
    int properlyParkedCars = 0;

    // Draw red for detected boxes that match reference boxes
    for (const auto& detectedBox : detectedBoxes) {
        bool isMatched = false;
        for (size_t i = 0; i < referenceBoxes.size(); ++i) {
            cv::Rect intersection = detectedBox & referenceBoxes[i];
            if (intersection.area()>0){
                isMatched = true;
                matched[i] = true; // Mark this reference box as matched
                
                cv::Rect bb = referenceBoxes[i];
                break;
            }
        }
        if (isMatched) {
            properlyParkedCars++; // Increment properly parked cars counter
        }
    }

    // Draw blue for reference boxes that are not detected
    for (size_t i = 0; i < referenceBoxes.size(); ++i) {
        
        if (!matched[i]) {
            rectangle(minimapPlot, referenceBoxes[i], emptyColor, cv::FILLED);
            emptySpaces++; // Increment empty spaces counter
        }
        else {
            rectangle(minimapPlot, referenceBoxes[i], carColor, cv::FILLED);
        }
    }

    // Optionally, draw the counters on the minimap image
    string emptySpacesText = "Empty Spaces: " + to_string(emptySpaces);
    string properlyParkedCarsText = "Properly Parked Cars: " + to_string(properlyParkedCars);
    
    putText(minimapPlot, emptySpacesText, Point(10, minimapPlot.rows - 200), FONT_HERSHEY_DUPLEX, 1.3, Scalar(0, 0, 0), 2);
    putText(minimapPlot, properlyParkedCarsText, Point(10, minimapPlot.rows - 100), FONT_HERSHEY_DUPLEX, 1.3, Scalar(0, 0, 0), 2);


    // Return the minimap plot
    return minimapPlot;
}

// Function to add an overlay of a minimap to an image
Mat imageWithMinimap(const Mat& img, const Mat& minimap) {

    Mat copyImg = img.clone();
    // Get dimensions of both images (resize minimap)
    double resizeCoeff = 0.3; // To be changed if needed
    int imgWidth = img.cols;
    int imgHeight = img.rows;
    int minimapWidth = static_cast<int>(resizeCoeff * minimap.cols);
    int minimapHeight = static_cast<int>(resizeCoeff * minimap.rows);
    Mat resizedMinimap;
    resize(minimap, resizedMinimap, cv::Size(minimapWidth, minimapHeight), 0, 0, cv::INTER_LINEAR);  // Resize the minimap


    // Define the position for the minimap
    int xPos = 20;
    int yPos = imgHeight - minimapHeight - 130;

    // Check if minimap fits inside the background
    if (minimapWidth > imgWidth || minimapHeight > imgHeight) {
        std::cerr << "Error: The minimap is too large!" << std::endl;
        cv::Mat errorMat(0, 0, CV_8UC1);
        return errorMat;
    }

    // Define the region of interest (ROI) on the background where the minimap will go
    cv::Rect roi(xPos, yPos, minimapWidth, minimapHeight);

    // Overlay the minimap onto the background image
    resizedMinimap.copyTo(copyImg(roi));
    
    return copyImg;
}
// ===================================




// ===================================
// FUNCTIONS FOR PARKING SPOTS DETECTION

// Function that returns parking spots
vector<Rect> parkingSpotsDetect(const Mat& img, double overlapThreshold){

    // convert to image from bgr to grayscale
    Mat gray, blurred, edges;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    // Apply GaussianBlur to reduce noise 
    GaussianBlur(gray, blurred, Size(3, 3), 0);
    
    // Apply Canny Edge Detection
    Canny(blurred, edges, 30, 120, 3);
    
    // Use Hough Line Transform to detect lines from edges
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 100, 50, 10);
    
    // if no lines detected
    if (lines.empty()) {  
        cout << "No lines were detected" << endl;
        return {};
    }

    // Sort end points of lines in X2 and Y2
    vector<Rect> rects;
    vector<int> X2, Y2;
    for (const Vec4i& line : lines) {
          int x1 = line[0], y1 = line[1];
          int x2 = line[2], y2 = line[3];

           X2.push_back(x2);
           Y2.push_back(y2);
    }
    sort(X2.begin(), X2.end());
    sort(Y2.begin(), Y2.end());

    // Loop through lines to find the closest line to form bounding boxes
    for (const Vec4i& line : lines) {
        int x1 = line[0], y1 = line[1];
        int x2 = line[2], y2 = line[3];
        
        if ((x2 - x1) > 10 && (y2-y1) > 10) { 
           
            // Compute the slant angle
            double angle = computeSlantAngle(x1, y1, x2, y2);
            
            int clearance = 10;  // clearance value
            if (angle> 5 && angle < 45) {  //ensure angle is between 5 and 45
                
                int t1 = x2; 
                int t2 = y2;
                for (int x : X2) {
                    if (x > x2) {
                        t1 = x;
                        break;
                    }
                }
                for (int y : Y2) {
                    if (y > y2) {
                        t2 = y;
                        break;
                    }
                }
                
                float centerX = (x1 + x2)/2;
                float centerY = (y1 + t2)/2;
                float width = t1-x1-clearance;
                float height= t2 - y1 + clearance;
                
                cv::Point2f center(centerX, centerY);

                cv::Size2f size(width, height);

                cv::Rect box = rotatedRectToBoundingRect(center, size, angle);  
       
                rects.push_back(box);

            }
        }
    }
    
    vector<Rect> mergedRectangles = mergeRectangles(rects, overlapThreshold);
    return mergedRectangles;

}

// Function to convert a rotated rectangle (defined by center, size, and angle) to an axis-aligned bounding rectangle
Rect rotatedRectToBoundingRect(Point2f center, Size2f size, float angle) {
    // Create a cv::RotatedRect from the center, size, and angle
    RotatedRect rotatedRect(center, size, angle);

    // Array to hold the four corner points of the rotated rectangle
    Point2f vertices[4];
    
    // Get the four corner points of the rotated rectangle
    rotatedRect.points(vertices);

    // Initialize the bounds of the axis-aligned bounding box
    int xMin = numeric_limits<int>::max();
    int yMin = numeric_limits<int>::max();
    int xMax = numeric_limits<int>::min();
    int yMax = numeric_limits<int>::min();

    // Compute the minimum and maximum x and y coordinates
    for (int i = 0; i < 4; ++i) {
        xMin = min(xMin, static_cast<int>(vertices[i].x));
        yMin = min(yMin, static_cast<int>(vertices[i].y));
        xMax = max(xMax, static_cast<int>(vertices[i].x));
        yMax = max(yMax, static_cast<int>(vertices[i].y));
    }

    // Return the axis-aligned bounding box
    return Rect(Point(xMin, yMin), Point(xMax, yMax));
}

// Function to merge rectangles based on overlap threshold
vector<Rect> mergeRectangles(const vector<Rect>& rectangles, double overlapThreshold) {
    vector<Rect> mergedRectangles;
    vector<bool> used(rectangles.size(), false);
    
    for (size_t i = 0; i < rectangles.size(); ++i) {
        if (used[i]) continue;
        Rect rect1 = rectangles[i];
        Rect merged = rect1;
        for (size_t j = i + 1; j < rectangles.size(); ++j) {
            if (used[j]) continue;
            Rect rect2 = rectangles[j];
            Rect interRect = computeIntersection(rect1, rect2);
            if (interRect.area() > 0) {
                int interArea = computeArea(interRect);
                int rect1Area = computeArea(rect1);
                int rect2Area = computeArea(rect2);
                int unionArea = rect1Area + rect2Area - interArea;
                double overlapRatio = (double)interArea / unionArea;
                if (overlapRatio >= overlapThreshold) {
                    merged = Rect(
                        min(merged.x, rect2.x),
                        min(merged.y, rect2.y),
                        max(merged.x + merged.width, rect2.x + rect2.width) - min(merged.x, rect2.x),
                        max(merged.y + merged.height, rect2.y + rect2.height) - min(merged.y, rect2.y)
                    );
                    used[j] = true;
                }
            }
        }
        mergedRectangles.push_back(merged);
    }
    
    return mergedRectangles;
}
// ===================================




// ===================================
// FUNCTIONS FOR CAR DETECTION

// Function to detect cars and return bounding boxes. Output: vector of rects (bounding boxes)
vector<Rect> carDetect(const Mat& img) {

    // Convert image to HSV color space
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    // Define HSV ranges for colors
    Scalar lower_white(0, 0, 200);
    Scalar upper_white(180, 25, 255);

    Scalar lower_blue(100, 150, 0);
    Scalar upper_blue(140, 255, 255);

    Scalar lower_red1(0, 150, 0);
    Scalar upper_red1(10, 255, 255);
    Scalar lower_red2(170, 150, 0);
    Scalar upper_red2(180, 255, 255);

    Scalar lower_black(0, 0, 0);
    Scalar upper_black(180, 255, 50);

    Scalar lower_ash(0, 0,150);
    Scalar upper_ash(180, 20, 255);

    // Create masks for each color
    Mat mask_white, mask_blue, mask_red1, mask_red2, mask_black,mask_ash;
    inRange(hsv, lower_white, upper_white, mask_white);
    inRange(hsv, lower_blue, upper_blue, mask_blue);
    inRange(hsv, lower_red1, upper_red1, mask_red1);
    inRange(hsv, lower_red2, upper_red2, mask_red2);
    inRange(hsv, lower_black, upper_black, mask_black);

    // Combine red masks
    Mat mask_red;
    bitwise_or(mask_red1, mask_red2, mask_red);

    // Combine all masks
    Mat combined_mask;
    bitwise_or(mask_white, mask_blue, combined_mask);
    bitwise_or(combined_mask, mask_red, combined_mask);
    bitwise_or(combined_mask, mask_black, combined_mask);

    // Apply morphological operations to refine the mask
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat opened, closed;
    morphologyEx(combined_mask, opened, MORPH_OPEN, kernel);
    morphologyEx(opened, closed, MORPH_CLOSE, kernel);

    // Find contours from the processed mask
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(closed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Define minimum area and aspect ratio thresholds
    const int min_area = 1000;
    const double min_aspect_ratio = 0.5;
    const double max_aspect_ratio = 10;

    // Vector to hold bounding boxes of detected cars
    vector<Rect> bounding_boxes;

    // Process contours to find bounding boxes
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > min_area) {
            Rect bounding_box = boundingRect(contour);
            double aspect_ratio = static_cast<double>(bounding_box.width) / bounding_box.height;
            if (aspect_ratio >= min_aspect_ratio && aspect_ratio <= max_aspect_ratio) {
                bounding_boxes.push_back(bounding_box);
            }
        }
    }
    return applyNMS(bounding_boxes,.1);

}

// Function to apply Non-Maximum Suppression
vector<Rect> applyNMS(const vector<Rect>& boxes, float threshold) {

    vector<Rect> result;
    vector<bool> selected(boxes.size(), true);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!selected[i]) continue;
        result.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (!selected[j]) continue;
            if (computeIoU(boxes[i], boxes[j]) > threshold) {
                selected[j] = false;
            }
        }
    }
    return result;

}

// Function to check if a parking space contains a car
vector<Rect> checkCarsInBB(const Mat& image, const vector<Rect>& boundingBoxes, double minCarArea) {

    Mat copyImage = image.clone();
    vector<Rect> detectedCars;
    for (const Rect& bbox : boundingBoxes) {
        if (bbox.x < 0 || bbox.y < 0 || bbox.x + bbox.width > image.cols || bbox.y + bbox.height > image.rows) {
            // Skip out-of-bounds rectangles
            continue;
        }

        // Extract the region of interest (image of parking space)
        Mat roi = copyImage(bbox).clone();
      
        // Apply car detection algorithm on the Roi
        vector<Rect> boundingBoxes_detect = carDetect(roi); // extract the bounding boxes of car detection
        
        // Check if there is a car detected
        bool found_non_zero_box = false;
        for (const auto& box : boundingBoxes_detect) {
            if (box.width > 0 && box.height > 0) {
                found_non_zero_box = true;
                break;
            }
        }
        if (found_non_zero_box) {
            detectedCars.push_back(bbox);
        }
    }
    return detectedCars;

}

vector<int> countProperParkings(const Mat& image, const vector<Rect>& detectedCars, const vector<Rect> detectedCarsInsideBB) {

    vector<int> proper_improperParkings;
    int properlyParkedCount = 0;
    int improperlyParkedCount = 0;
    // Draw bounding boxes on the input image
    for (const auto& bbox : detectedCars) {

        int bboxWidth = bbox.width;
        int bboxHeight = bbox.height;
        int area =bboxWidth*bboxHeight;

        // check for proper parking using reference bounding boxes and the box detected
        bool properlyParked = isProperlyParked(bbox, detectedCarsInsideBB);  

        if (properlyParked) {
            properlyParkedCount++;
        }
        else{
            improperlyParkedCount++;
        }
        proper_improperParkings.push_back(properlyParkedCount);
        proper_improperParkings.push_back(improperlyParkedCount);   
    }
    return proper_improperParkings;

}
// ===================================




// ===================================
// FUNCTIONS FOR CAR SEGMENTATION

// Function to segment cars using the absolute difference between 2 images
Mat segmentCars(const Mat& image1, const Mat& image2) {

    if (image1.empty() || image2.empty() || image1.size() != image2.size() || image1.type() != image2.type()) {
        cerr << "Error: Images are empty or have different sizes/types!" << endl;
        return Mat();
    }

    Mat gauss, imageDiff, copy1 = image1.clone();

    // Compute absolute difference
    absdiff(image1, image2, imageDiff);

    // Convert to grayscale for better segmentation
    Mat gray;
    cvtColor(imageDiff, gray, COLOR_BGR2GRAY);

    // Apply Gaussian blur to reduce noise
    GaussianBlur(gray, gauss, Size(7, 7), 0);

    // Threshold the image to create a binary mask
    Mat binary;
    threshold(gauss, binary, 30, 255, THRESH_BINARY);

    // Iterate through the image and highlight the segmented cars
    for (int y = 0; y < image1.rows; y++) {
        for (int x = 0; x < image1.cols; x++) {
            if (binary.at<uchar>(y, x) > 0) { // If there is a difference
                copy1.at<Vec3b>(y, x)[2] = min(image1.at<Vec3b>(y, x)[2] + 200, 255); // Increase Red channel
            }
        }
    }

    return copy1;

}

// function that colour ground truth image based on mask
Mat colorizeImageWithMask(Mat& image, const Mat& maskImage) {

    Mat mainImage = image.clone();
    // Ensure the main image is in color (3 channels) and mask image is grayscale (1 channel)
    if (mainImage.channels() != 3 || maskImage.channels() != 1) {
        throw std::invalid_argument("Images must be 3 channels (main image) and 1 channel (mask image).");
    }

    // Check that the dimensions of both images match
    if (mainImage.size() != maskImage.size()) {
        throw std::invalid_argument("Images must have the same dimension.");
    }

    // Iterate over each pixel
    for (int y = 0; y < mainImage.rows; ++y) {
        for (int x = 0; x < mainImage.cols; ++x) {
            uchar maskValue = maskImage.at<uchar>(y, x);

            // Get a reference to the current pixel in the main image
            cv::Vec3b& pixel = mainImage.at<cv::Vec3b>(y, x);

            // Colorize based on mask value
            switch (maskValue) {
                case 1:
                    pixel = cv::Vec3b(255, 0, 0); // Blue
                    break;
                case 2:
                    pixel = cv::Vec3b(0, 0, 255); // Red
                    break;
                default:
                    // Keep the original color for mask values 0 or any other values
                    break;
            }
        }
    }
    return mainImage;

}
// ===================================




// ===================================
// HELPER FUNCTIONS TO OTHER FUNCTIONS

// Function to compute slant angle of a line
double computeSlantAngle(int x1, int y1, int x2, int y2) {

    double dx = x2 - x1;
    double dy = y2 - y1;
    double angleRad = atan2(dy, dx);
    double angleDeg = angleRad * (180.0 / CV_PI);
    return angleDeg;

}

// Function to compute area of a rectangle
int computeArea(const Rect& rect) {

    return rect.width * rect.height;

}

// Function to compute intersection of two rectangles
Rect computeIntersection(const Rect& rect1, const Rect& rect2) {

    int x1 = max(rect1.x, rect2.x);
    int y1 = max(rect1.y, rect2.y);
    int x2 = min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = min(rect1.y + rect1.height, rect2.y + rect2.height);
    
    if (x1 < x2 && y1 < y2) {
        return Rect(x1, y1, x2 - x1, y2 - y1);
    } else {
        return Rect(); // Return empty rectangle
    }

}

// function to check if a car is properly parked
bool isProperlyParked(const Rect& testBox, const vector<Rect>& referenceBoxes) {
   
    for (const auto& refBox : referenceBoxes) { //loop through detected parking spaces
        
        int thresh = 800; // Distance threshold
        // Compute centers of bounding boxes
        Point center1(refBox.x + refBox.width / 2, refBox.y + refBox.height / 2);
        Point center2(testBox.x + testBox.width / 2, testBox.y + testBox.height / 2);
       
        // Check if distance between centers is within minimum threshold
        if (norm(center1 - center2) < thresh){
            return true;
        }
    }
    return false;
}

// Function to extract BB from mask image
vector<Rect> extractBBMask(const Mat& mask) {

    // Extracting bounding boxes from mask
    std::vector<cv::Rect> allBoundingBoxes;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    // Extracting groundtruth car boundingboxes from mask
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // Loop through the contours to get all bounding boxes
    for (size_t i = 0; i < contours.size(); ++i) {
        Rect boundingBox = boundingRect(contours[i]);
        allBoundingBoxes.push_back(boundingBox);
    }
    return allBoundingBoxes;
}
// ===================================




// ===================================
// METRICS EVALUATION FUNCTIONS

// Function to calculate Intersection over Union (IoU) between two bounding boxes
float computeIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    int xA = std::max(boxA.x, boxB.x);
    int yA = std::max(boxA.y, boxB.y);
    int xB = std::min(boxA.x + boxA.width, boxB.x + boxB.width);
    int yB = std::min(boxA.y + boxA.height, boxB.y + boxB.height);

    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    int boxAArea = boxA.width * boxA.height;
    int boxBArea = boxB.width * boxB.height;

    return (float)interArea / (boxAArea + boxBArea - interArea);
}

// Function to calculate mean Intersection over Union (mIoU)
float computeMeanIoU(const std::vector<cv::Rect>& gtBoxes, const std::vector<cv::Rect>& predBoxes) {
    if (gtBoxes.empty() || predBoxes.empty()) return 0.0f;

    std::vector<float> ious;

    for (const auto& gtBox : gtBoxes) {
        for (const auto& predBox : predBoxes) {
            float iou = computeIoU(gtBox, predBox);
            if (iou>0.1){
            ious.push_back(iou);
            }
        }
    }

    if (ious.empty()) return 0.0f;

    float sumIoU = 0.0f;
    for (const auto& iou : ious) {
        if (iou>0.1){
        sumIoU += iou;
        }
    }

    return sumIoU / ious.size();
}

// Function to compute the mean Average Precision (mAP)
float computeMeanAP(const std::vector<std::vector<cv::Rect>>& allGtBoxes, const std::vector<std::vector<cv::Rect>>& allPredBoxes, float iouThreshold) {
    if (allGtBoxes.size() != allPredBoxes.size()) return 0.0f;

    float sumAP = 0.0f;
    for (size_t i = 0; i < allGtBoxes.size(); ++i) {
        float ap = computeAveragePrecision(allGtBoxes[i], allPredBoxes[i], iouThreshold);
        sumAP += ap;
    }

    return sumAP / allGtBoxes.size();
}

// Function to calculate Average Precision (AP)
float computeAveragePrecision(const vector<cv::Rect>& gtBoxes, const vector<cv::Rect>& predBoxes, float iouThreshold) {
    std::vector<float> ious;

    // Compute IoUs between all pairs
    for (const auto& gtBox : gtBoxes) {
        for (const auto& predBox : predBoxes) {
            float iou = computeIoU(gtBox, predBox);
            if (iou>=iouThreshold){
            ious.push_back(iou);
            }
        }
    }

    // Sort IoUs in descending order
    sort(ious.begin(), ious.end(), greater<float>());

    // Calculate precision and recall
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = gtBoxes.size();

    for (const auto& iou : ious) {
        if (iou >= iouThreshold) {
            truePositives++;
            falseNegatives--;
        } else {
            falsePositives++;
        }
    }

    if (truePositives == 0) return 0.0f;

    float precision = (float)truePositives / (truePositives + falsePositives);
    float recall = (float)truePositives / (truePositives + falseNegatives);

    // Calculate Average Precision
    return precision * recall / (precision + recall);
}
// ===================================