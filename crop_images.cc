#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// #include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
using namespace cv;

const int LOOP_NUM = 10;
const int GOOD_PTS_MAX = 50;
const float GOOD_PORTION = 0.15f;

template<class KPDetector>
struct SURFDetector
{
    KPDetector surf;
    SURFDetector(double hessian = 800.0)
        :surf(hessian)
    {
    }
    template<class T>
    void operator()(const T& in, const T& mask, vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        surf(in, mask, pts, descriptors, useProvided);
    }
};

void resetROI(const cv::Rect& roi, cv::Rect& roi_reset, const float* span_aspects) {
  int center_x = roi.x + roi.width / 2;
  int center_y = roi.y + roi.height / 2;
  roi_reset.x = center_x - span_aspects[0] * roi.width; // left x
  roi_reset.y = center_y - span_aspects[1] * roi.height; // top y
  roi_reset.width = (span_aspects[0] + span_aspects[2]) * roi.width; // width 
  roi_reset.height = (span_aspects[1] + span_aspects[3]) * roi.height; // height
}

void scaleROI(const cv::Rect& roi, cv::Rect& roi_scaled, const float& scale) {
  roi_scaled.x = roi.x - roi.width * (scale - 1) / 2;
  roi_scaled.y = roi.y - roi.height * (scale - 1) / 2; 
  roi_scaled.width = scale * roi.width;
  roi_scaled.height = scale * roi.height;
}

void regROI(cv::Rect& roi, int width, int height) {
  roi.x = MAX(0, roi.x);
  roi.y = MAX(0, roi.y);
  roi.width = MIN(roi.width, width - roi.x);
  roi.height = MIN(roi.height, height - roi.y);
}

static Mat drawGoodMatches(
    const Mat& cpu_img1,
    const Mat& cpu_img2,
    const vector<KeyPoint>& keypoints1,
    const vector<KeyPoint>& keypoints2,
    vector<DMatch>& matches,
    vector<Point2f>& scene_corners_
)
{
    //-- Sort matches and preserve top 10% matches
    std::sort(matches.begin(), matches.end());
    std::vector< DMatch > good_matches;
    double minDist = matches.front().distance,
           maxDist = matches.back().distance;

    const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
    for( int i = 0; i < ptsPairs; i++ )
    {
        good_matches.push_back( matches[i] );
    }
    std::cout << "\nMax distance: " << maxDist << std::endl;
    std::cout << "Min distance: " << minDist << std::endl;

    std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;

    // drawing the results
    Mat img_matches;
    drawMatches( cpu_img1, keypoints1, cpu_img2, keypoints2,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  );

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( cpu_img1.cols, 0 );
    obj_corners[2] = cvPoint( cpu_img1.cols, cpu_img1.rows );
    obj_corners[3] = cvPoint( 0, cpu_img1.rows );
    std::vector<Point2f> scene_corners(4);

    Mat H = findHomography( obj, scene, CV_RANSAC );
    perspectiveTransform( obj_corners, scene_corners, H);

    scene_corners_ = scene_corners;

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches,
          scene_corners[0] + Point2f( (float)cpu_img1.cols, 0), scene_corners[1] + Point2f( (float)cpu_img1.cols, 0),
          Scalar( 0, 255, 0), 2, CV_AA );
    line( img_matches,
          scene_corners[1] + Point2f( (float)cpu_img1.cols, 0), scene_corners[2] + Point2f( (float)cpu_img1.cols, 0),
          Scalar( 0, 255, 0), 2, CV_AA );
    line( img_matches,
          scene_corners[2] + Point2f( (float)cpu_img1.cols, 0), scene_corners[3] + Point2f( (float)cpu_img1.cols, 0),
          Scalar( 0, 255, 0), 2, CV_AA );
    line( img_matches,
          scene_corners[3] + Point2f( (float)cpu_img1.cols, 0), scene_corners[0] + Point2f( (float)cpu_img1.cols, 0),
          Scalar( 0, 255, 0), 2, CV_AA );
    return img_matches;
}

int main(int argc, char** argv) {

std::ifstream fin_gt("../logo/BelgaLogos/qset3_internal_and_local.gt");

std::string instance_name, logo_name, image_name, instance_type;
int instance_state, X1, Y1, X2, Y2;
std::string path_logo_folder;
std::string logo_name_current = "";
while (fin_gt >> instance_name >> logo_name >> image_name >> instance_type 
              >> instance_state >> X1 >> Y1 >> X2 >> Y2) {

if (logo_name != logo_name_current) { // create directory
   path_logo_folder = "../logo/BelgaLogos/logos/" + logo_name;
   boost::filesystem::create_directory(path_logo_folder);
   logo_name_current = logo_name;
}

if (instance_state == 1) { // contains current logo
  std::string img_path = "../logo/BelgaLogos/images/" + image_name;
  std::cout << img_path << std::endl;
  cv::Mat img = cv::imread(img_path);
  // cv::imshow("img", img);
  // cv::waitKey(0);

  // crop image
  if (X2 < X1 || Y2 < Y1)
    continue;

  cv::Rect roi = cv::Rect(X1, Y1, X2-X1+1, Y2-Y1+1);
  scaleROI(roi, roi, 1.5);
  regROI(roi, img.cols, img.rows);

  cv::Mat img_roi = img(roi);
  cv::Mat img_roi_gray;
  cv::cvtColor(img_roi, img_roi_gray, CV_BGR2GRAY);
  // SURF detector
  int minHessian = 800;
  SURFDetector<SURF>     cpp_surf;
  // SurfFeatureDetector detector( minHessian );
  
  std::vector<KeyPoint> keypoints;
  cv::Mat descriptorsCPU;
  // detector.detect( img_roi_gray, keypoints);
  cpp_surf(img_roi_gray, Mat(), keypoints, descriptorsCPU);
  std::cout << descriptorsCPU.cols << " " << descriptorsCPU.rows << std::endl;
  //-- Draw keypoints
  Mat img_keypoints;
  cv::drawKeypoints( img_roi_gray, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  //-- Show detected (drawn) keypoints
  cv::imshow("Keypoints", img_keypoints );
  cv::waitKey(0);
  std::string path_logo = path_logo_folder + "/" + image_name;
  cv::imwrite(path_logo, img_roi);
}
}
}
