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

    cvtColor( img, img_gray, COLOR_BGR2GRAY );

    bilateralFilter( img_gray, blurred, 5, 10, 2.5 );
    GaussianBlur( blurred, blurred, Size(5, 5), 2, 2 );

    Canny( blurred, edges, 70, 150 );
    imshow( "edges", edges );

    std::vector<cv::Vec4i> lines;
    HoughLinesP( edges, lines, 1, CV_PI / 180, 20, 50, 10 );

    // Crea un'immagine per visualizzare le linee
    Mat lineImage = img.clone();
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(lineImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
    }

    namedWindow( "Figure" );
    imshow( "Figure", lineImage );
    waitKey( 0 );

    return 0;

}

