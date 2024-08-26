#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>

using namespace cv;

Mat img2, detected_edges, img3, img_gray;

int main( int argc, char** argv ){

    Mat img = imread( argv[1] );
    Mat img_gray, blurred, edges;
    std::vector<RotatedRect> parkingSpaces;

    cvtColor( img, img_gray, COLOR_BGR2GRAY );

    //threshold( img_gray, img_gray, 190, 255, THRESH_BINARY );


    bilateralFilter( img_gray, blurred, 5, 10, 2.5 );
    GaussianBlur( blurred, blurred, Size(5, 5), 2, 2 );

    adaptiveThreshold(blurred, blurred, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 3, 2);

    Canny( blurred, edges, 70, 150 );
    imshow( "edges", edges );

    Mat morph;
    morphologyEx(edges, edges, MORPH_CLOSE, Mat::ones(5, 5, CV_8U));

    std::vector<cv::Vec4i> lines;
    HoughLinesP( edges, lines, 1, CV_PI / 180, 20, 30, 10 );
    /*
    // Crea un'immagine per visualizzare le linee
    Mat lineImage = img.clone();
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(lineImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
    }
    */
    // PROVA
    // 7. Creazione di una maschera basata sulle linee rilevate
    Mat lineImage = Mat::zeros(img.size(), CV_8UC1);
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(lineImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 2, LINE_AA);
    }

    imshow( "lineImage1", lineImage );

    // 8. Rilevazione dei contorni nella maschera di linee
    std::vector<std::vector<Point>> contours;
    findContours(lineImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

     // 9. Filtraggio dei contorni per ottenere spazi di parcheggio ragionevoli
    for (const auto &contour : contours) {
        RotatedRect rect = minAreaRect(contour);
        if (rect.size.width > 25 && rect.size.height > 25) {
            parkingSpaces.push_back(rect);
        }
    }

    // Visualizzazione degli spazi rilevati
    for (const auto &space : parkingSpaces) {
        Point2f points[4];
        space.points(points);
        for (int i = 0; i < 4; ++i)
            line(img, points[i], points[(i + 1) % 4], Scalar(0, 255, 0), 2);
    }

    imshow("Detected Parking Spaces", img);
    waitKey(0);

    // FINE PROVA
    //namedWindow( "Figure" );
    //imshow( "Figure", lineImage );
    //waitKey( 0 );

    return 0;

}
