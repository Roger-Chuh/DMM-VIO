#include "EDLib.h"
#include "edge_drawing.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
namespace dso {
namespace ED {
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
  const char* filename;
  if (argc > 1)
    filename = argv[1];
  else
    filename =
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "planet_glow.jpg";

  Mat testImg, ellipsImg0, ellipsImg1;
  Mat colorImg = imread(filename);
  cvtColor(colorImg, testImg, COLOR_BGR2GRAY);

  Ptr<dso::ED::EdgeDrawing> ed = dso::ED::createEdgeDrawing();
  vector<Vec6d> ellipses;
  vector<Vec4f> lines;

  TickMeter tm;
  for (int i = 0; i < 3; i++) {
    cout << "\n#################################################";
    cout << "\n####### ( " << i << " ) ORIGINAL & OPENCV COMPARISON ######";
    cout << "\n#################################################\n";

    ed->params.EdgeDetectionOperator = dso::ED::EdgeDrawing::SOBEL;
    ed->params.GradientThresholdValue = 36;
    ed->params.AnchorThresholdValue = 8;
    ed->params.Sigma = 1.0;

    // Detection of edge segments from an input image
    tm.start();
    // Call ED constructor
    ED testED = ED(testImg, SOBEL_OPERATOR, 36, 8, 1, 10, 1.0, true);
    tm.stop();
    std::cout << "testED.getEdgeImage()  (Original)  : " << tm.getTimeMilli() << " ms." << endl;

    tm.reset();
    tm.start();
    ed->detectEdges(testImg);
    tm.stop();
    std::cout << "detectEdges()            (OpenCV)  : " << tm.getTimeMilli() << " ms." << endl;

    Mat anchImg = testED.getAnchorImage();
    Mat gradImg = testED.getGradImage();
    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "GradImage.png",
        gradImg);
    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "AnchorImage.png",
        anchImg);

    Mat edgeImg1, diff;
    Mat edgeImg0 = testED.getEdgeImage();
    ed->getEdgeImage(edgeImg1);
    absdiff(edgeImg0, edgeImg1, diff);
    cout << "different pixel count              : " << countNonZero(diff) << endl;

    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "EdgeImageImpl.png",
        edgeImg0);
    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "EdgeImageOpenCV.png",
        edgeImg1);

    //***************************** EDLINES Line Segment Detection
    //***************************** Detection of lines segments from edge
    // segments instead of input image Therefore, redundant detection of edge
    // segmens can be avoided
    tm.reset();
    tm.start();
    EDLines testEDLines = EDLines(testED);
    tm.stop();
    cout << "-------------------------------------------------\n";
    cout << "testEDLines.getLineImage()         : " << tm.getTimeMilli() << " ms." << endl;
    Mat lineImg0 = testEDLines.getLineImage();  // draws on an empty image

    tm.reset();
    tm.start();
    ed->detectLines(lines);
    tm.stop();
    cout << "detectLines()            (OpenCV)  : " << tm.getTimeMilli() << " ms." << endl;

    Mat lineImg1 = Mat(lineImg0.rows, lineImg0.cols, CV_8UC1, Scalar(255));

    for (int i = 0; i < lines.size(); i++)
      line(lineImg1, Point2d(lines[i][0], lines[i][1]), Point2d(lines[i][2], lines[i][3]), Scalar(0), 1, LINE_AA);

    absdiff(lineImg0, lineImg1, diff);
    cout << "different pixel count              : " << countNonZero(diff) << endl;
    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "LinesImage.png",
        lineImg1);

    //***************************** EDCIRCLES Circle Segment Detection
    //***************************** Detection of circles from already available
    // EDPF or ED image
    tm.reset();
    tm.start();
    EDCircles testEDCircles = EDCircles(testEDLines);
    tm.stop();
    cout << "-------------------------------------------------\n";
    cout << "EDCircles(testEDLines)             : " << tm.getTimeMilli() << " ms." << endl;

    tm.reset();
    tm.start();
    ed->detectEllipses(ellipses);
    tm.stop();
    cout << "detectEllipses()         (OpenCV)  : " << tm.getTimeMilli() << " ms." << endl;
    cout << "-------------------------------------------------\n";

    vector<mCircle> found_circles = testEDCircles.getCircles();
    vector<mEllipse> found_ellipses = testEDCircles.getEllipses();

    cvtColor(testImg, ellipsImg0, COLOR_GRAY2BGR);
    cvtColor(testImg, ellipsImg1, COLOR_GRAY2BGR);

    for (int i = 0; i < found_circles.size(); i++) {
      Point center((int)found_circles[i].center.x, (int)found_circles[i].center.y);
      Size axes((int)found_circles[i].r, (int)found_circles[i].r);
      double angle(0.0);
      Scalar color = Scalar(0, 255, 0);

      ellipse(ellipsImg0, center, axes, angle, 0, 360, color, 1, LINE_AA);
    }

    for (int i = 0; i < found_ellipses.size(); i++) {
      Point center((int)found_ellipses[i].center.x, (int)found_ellipses[i].center.y);
      Size axes((int)found_ellipses[i].axes.width, (int)found_ellipses[i].axes.height);
      double angle = found_ellipses[i].theta * 180 / CV_PI;
      Scalar color = Scalar(255, 255, 0);

      ellipse(ellipsImg0, center, axes, angle, 0, 360, color, 1, LINE_AA);
    }

    for (size_t i = 0; i < ellipses.size(); i++) {
      Point center((int)ellipses[i][0], (int)ellipses[i][1]);
      Size axes((int)ellipses[i][2] + (int)ellipses[i][3], (int)ellipses[i][2] + (int)ellipses[i][4]);
      double angle(ellipses[i][5]);
      Scalar color = ellipses[i][2] == 0 ? Scalar(255, 255, 0) : Scalar(0, 255, 0);

      ellipse(ellipsImg1, center, axes, angle, 0, 360, color, 1, LINE_AA);
    }

    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "EllipsImage-Original.png",
        ellipsImg0);
    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "EllipsImage-OpenCV.png",
        ellipsImg1);

    //************************** EDPF Parameter-free Edge Segment Detection
    //**************************
    // Detection of edge segments with parameter free ED (EDPF)
    tm.reset();
    tm.start();
    EDPF testEDPF = EDPF(testImg);
    tm.stop();
    cout << "testEDPF.getEdgeImage()            : " << tm.getTimeMilli() << " ms." << endl;

    ed->params.EdgeDetectionOperator = EdgeDrawing::PREWITT;
    ed->params.GradientThresholdValue = 11;
    ed->params.AnchorThresholdValue = 3;
    ed->params.PFmode = true;

    tm.reset();
    tm.start();
    ed->detectEdges(testImg);
    tm.stop();
    std::cout << "detectEdges()  PF        (OpenCV)  : " << tm.getTimeMilli() << " ms." << endl;

    edgeImg0 = testEDPF.getEdgeImage();
    ed->getEdgeImage(edgeImg1);
    absdiff(edgeImg0, edgeImg1, diff);
    cout << "different pixel count              : " << countNonZero(diff) << endl;
    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "EdgeImage-PF-Impl.png",
        edgeImg0);
    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "EdgeImage-PF-OpenCV.png",
        edgeImg1);

    //*********************** EDCOLOR Edge Segment Detection from Color Images
    //**********************

    tm.reset();
    tm.start();
    EDColor testEDColor = EDColor(colorImg, 36);
    std::cout << "aa" << std::endl;
    edgeImg0 = testEDColor.getEdgeImage();
    std::cout << "bb" << std::endl;
    tm.stop();
    cout << "-------------------------------------------------\n";
    cout << "testEDColor.getEdgeImage()         : " << tm.getTimeMilli() << " ms." << endl;

    ed->params.EdgeDetectionOperator = EdgeDrawing::PREWITT;
    ed->params.GradientThresholdValue = 36;
    ed->params.AnchorThresholdValue = 4;
    ed->params.Sigma = 1.5;
    ed->params.PFmode = false;
    tm.reset();
    tm.start();
    // detectEdges 仅支持 CV_8UC1 灰度图，彩色图需先转换
    Mat colorImgGray;
    cvtColor(colorImg, colorImgGray, COLOR_BGR2GRAY);
    ed->detectEdges(colorImgGray);
    tm.stop();
    cout << "detectEdges()            (OpenCV)  : " << tm.getTimeMilli() << " ms." << endl;

    ed->getEdgeImage(edgeImg1);
    absdiff(edgeImg0, edgeImg1, diff);
    cout << "different pixel count              : " << countNonZero(diff) << endl;
    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "EdgeImage-Color.png",
        edgeImg0);

    tm.reset();
    tm.start();
    // get lines from color image
    EDLines colorLine = EDLines(testEDColor);
    tm.stop();
    cout << "-------------------------------------------------\n";
    cout << "get lines from color image         : " << tm.getTimeMilli() << " ms." << endl;
    lineImg0 = colorLine.getLineImage();
    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "LinesImage-Color.png",
        lineImg0);
    tm.reset();
    tm.start();
    ed->detectLines(lines);
    tm.stop();
    cout << "detectLines()            (OpenCV)  : " << tm.getTimeMilli() << " ms." << endl;

    lineImg1 = Mat(lineImg0.rows, lineImg0.cols, CV_8UC1, Scalar(255));

    for (int i = 0; i < lines.size(); i++)
      line(lineImg1, Point2d(lines[i][0], lines[i][1]), Point2d(lines[i][2], lines[i][3]), Scalar(0), 1, LINE_AA);

    absdiff(lineImg0, lineImg1, diff);
    cout << "different pixel count              : " << countNonZero(diff) << endl;
    cout << "-------------------------------------------------\n";
    tm.reset();
    tm.start();
    // get circles from color image
    EDCircles colorCircle = EDCircles(testEDColor);
    tm.stop();
    cout << "get circles from color image       : " << tm.getTimeMilli() << " ms." << endl;
    tm.reset();
    tm.start();
    ed->detectEllipses(ellipses);
    tm.stop();
    cout << "detectEllipses()         (OpenCV)  : " << tm.getTimeMilli() << " ms." << endl;

    found_circles = colorCircle.getCircles();
    found_ellipses = colorCircle.getEllipses();

    ellipsImg0 = colorImg.clone();
    ellipsImg1 = colorImg.clone();
    for (int i = 0; i < found_circles.size(); i++) {
      Point center((int)found_circles[i].center.x, (int)found_circles[i].center.y);
      Size axes((int)found_circles[i].r, (int)found_circles[i].r);
      double angle(0.0);
      Scalar color = Scalar(0, 255, 0);

      ellipse(ellipsImg0, center, axes, angle, 0, 360, color, 2, LINE_AA);
    }

    for (int i = 0; i < found_ellipses.size(); i++) {
      Point center((int)found_ellipses[i].center.x, (int)found_ellipses[i].center.y);
      Size axes((int)found_ellipses[i].axes.width, (int)found_ellipses[i].axes.height);
      double angle = found_ellipses[i].theta * 180 / CV_PI;
      Scalar color = Scalar(255, 255, 0);

      ellipse(ellipsImg0, center, axes, angle, 0, 360, color, 2, LINE_AA);
    }

    for (size_t i = 0; i < ellipses.size(); i++) {
      Point center((int)ellipses[i][0], (int)ellipses[i][1]);
      Size axes((int)ellipses[i][2] + (int)ellipses[i][3], (int)ellipses[i][2] + (int)ellipses[i][4]);
      double angle(ellipses[i][5]);
      Scalar color = ellipses[i][2] == 0 ? Scalar(255, 255, 0) : Scalar(0, 255, 0);

      ellipse(ellipsImg1, center, axes, angle, 0, 360, color, 2, LINE_AA);
    }

    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "EllipsImageColor-Original.png",
        ellipsImg0);
    imwrite(
        "/home/roger/work/dm-vio/dm-vio/src/dso/FullSystem/ED_Lib/"
        "EllipsImageColor-OpenCV.png",
        ellipsImg1);
  }
  return 0;
}
}  // namespace ED
}  // namespace dso

// 全局入口：链接器要求可执行文件具有全局 ::main，这里转发到 dso::ED::main。
int main(int argc, char** argv) { return dso::ED::main(argc, argv); }