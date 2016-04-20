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
#include "opencv2/calib3d/calib3d.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "directory.h"

using namespace cv;
using namespace std;

int thresh = 100;
int max_thresh = 255;

int maxCorners = 100;
int maxTrackbar = 100;

int clusterCount_local = 5;
int clusterCount_global = 100;
RNG rng(12345);

// string imgfolder = "../logo/BelgaLogos/logos"; // 
string imgfolder = "../logo/tmplogo";

void cornerHarris_demo(const cv::Mat& src_gray, cv::Mat& target, std::vector<cv::Point>& keypoints)
{

  Mat dst, dst_norm, dst_norm_scaled;
  dst = Mat::zeros( src_gray.size(), CV_32FC1 );

  /// Detector parameters
  int blockSize = 5;
  int apertureSize = 3;
  double k = 0.04;

  /// Detecting corners
  cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

  /// Normalizing
  normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  // convertScaleAbs( dst_norm, target );

  /// Drawing a circle around corners
  for( int j = 0; j < dst_norm.rows ; j++ )
     { for( int i = 0; i < dst_norm.cols; i++ )
          {
            if( (int) dst_norm.at<float>(j,i) > thresh )
              {
               circle( target, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
               keypoints.push_back(Point(i, j));
              }
          }
     }
}

void goodFeaturesToTrack_Demo(const cv::Mat src_gray, cv::Mat& target, std::vector<cv::Point>& keypoints)
{

  /// Parameters for Shi-Tomasi algorithm
  vector<Point2f> corners;
  double qualityLevel = 0.01;
  double minDistance = 10;
  int blockSize = 5;
  bool useHarrisDetector = true;
  double k = 0.04;

  /// Apply corner detection
  goodFeaturesToTrack( src_gray,
               corners,
               maxCorners,
               qualityLevel,
               minDistance,
               Mat(),
               blockSize,
               useHarrisDetector,
               k );


  /// Draw corners detected
  std::cout<<"** Number of corners detected: "<<corners.size()<<std::endl;
  int r = 4;
  for( int i = 0; i < corners.size(); i++ )
     { circle( target, corners[i], r, Scalar(rng.uniform(0,255), rng.uniform(0,255),
              rng.uniform(0,255)), -1, 8, 0 ); 
       keypoints.push_back(corners[i]); 
     }
}

void cvtKeyPoints(const std::vector<cv::Point>& keypoints_pos, std::vector<KeyPoint>& keypoints) {
  keypoints.resize(keypoints_pos.size());
}

void surfFeature(const cv::Mat src, std::vector<cv::KeyPoint>& keypoints, cv::Mat& features) {
  SurfDescriptorExtractor extractor;
  extractor.compute(src, keypoints, features);
}


bool refineMatchesWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints,    
	const std::vector<cv::KeyPoint>& trainKeypoints,     
	float reprojectionThreshold,    
	std::vector<cv::DMatch>& matches,    
	cv::Mat& homography  )  
{  
	const int minNumberMatchesAllowed = 4;    
	if (matches.size() < minNumberMatchesAllowed)    
		return false;    
	// Prepare data for cv::findHomography    
	std::vector<cv::Point2f> queryPoints(matches.size());    
	std::vector<cv::Point2f> trainPoints(matches.size());    
	for (size_t i = 0; i < matches.size(); i++)    
	{    
		queryPoints[i] = queryKeypoints[matches[i].queryIdx].pt;    
		trainPoints[i] = trainKeypoints[matches[i].trainIdx].pt;    
	}    
	// Find homography matrix and get inliers mask    
	std::vector<unsigned char> inliersMask(matches.size());    
	homography = cv::findHomography(queryPoints,     
		trainPoints,     
		CV_FM_RANSAC,     
		reprojectionThreshold,     
		inliersMask);    
	std::vector<cv::DMatch> inliers;    
	for (size_t i=0; i<inliersMask.size(); i++)    
	{    
		if (inliersMask[i])    
			inliers.push_back(matches[i]);    
	}    
	matches.swap(inliers);  
        /*
	Mat homoShow;  
	drawMatches(src,queryKeypoints,frameImg,trainKeypoints,matches,homoShow,Scalar::all(-1),CV_RGB(255,255,255),Mat(),2);       
	imshow("homoShow",homoShow);   
        */
	return matches.size() > minNumberMatchesAllowed;   

}  


bool matchingDescriptor(const vector<KeyPoint>& queryKeyPoints,const vector<KeyPoint>& trainKeyPoints,  
	const Mat& queryDescriptors,const Mat& trainDescriptors,   
	Ptr<DescriptorMatcher>& descriptorMatcher, std::vector<DMatch>& m_Matches, 
        bool enableRatioTest = true)  
{  
	vector<vector<DMatch>> m_knnMatches;  
	 

	if (enableRatioTest)  
	{  
		cout<<"KNN Matching"<<endl;  
		const float minRatio = 1.2f / 1.5f;  
		descriptorMatcher->knnMatch(queryDescriptors,trainDescriptors,m_knnMatches,2);  
		for (size_t i=0; i<m_knnMatches.size(); i++)  
		{  
			const cv::DMatch& bestMatch = m_knnMatches[i][0];  
			const cv::DMatch& betterMatch = m_knnMatches[i][1];  
			float distanceRatio = bestMatch.distance / betterMatch.distance;  
			if (distanceRatio < minRatio)  
			{  
				m_Matches.push_back(bestMatch);  
			}  
		}  

	}  
	else  
	{  
		cout<<"Cross-Check"<<endl;  
		Ptr<cv::DescriptorMatcher> BFMatcher(new cv::BFMatcher(cv::NORM_HAMMING, true));  
		BFMatcher->match(queryDescriptors,trainDescriptors, m_Matches );  
	}  
	Mat homo;  
	float homographyReprojectionThreshold = 0.1;  
	bool homographyFound = refineMatchesWithHomography(  
		queryKeyPoints,trainKeyPoints,homographyReprojectionThreshold,m_Matches,homo);  

	if (!homographyFound)  
		return false;  
	else  
	{  
/*
		if (m_Matches.size()>10)
		{
			std::vector<Point2f> obj_corners(4);
			obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( src.cols, 0 );
			obj_corners[2] = cvPoint( src.cols, src.rows ); obj_corners[3] = cvPoint( 0, src.rows );
			std::vector<Point2f> scene_corners(4);
			perspectiveTransform( obj_corners, scene_corners, homo);
			line(frameImg,scene_corners[0],scene_corners[1],CV_RGB(255,0,0),2);  
			line(frameImg,scene_corners[1],scene_corners[2],CV_RGB(255,0,0),2);  
			line(frameImg,scene_corners[2],scene_corners[3],CV_RGB(255,0,0),2);  
			line(frameImg,scene_corners[3],scene_corners[0],CV_RGB(255,0,0),2);  
			return true;  
		}
*/
		return true;
	}  


}  

int main(int argc, char** argv) {
  std::vector<string> folders;
  fileutil::GetDir(imgfolder, &folders);
  std::cout << "Number of folders: " << folders.size() << std::endl;
  
  cv::Mat features_all_classes;
  vector<int> idx_all_classes;
  for (int i = 0; i < folders.size(); ++i) {
    std::vector<string> files;
    string folder = imgfolder + "/" + folders[i]; 
    std::cout << folder << std::endl;
    fileutil::GetDirRecursive(folder, &files);
    std::cout << "Number of images: " << files.size() << std::endl;
    /* 
    for (int k = 0; k < files.size(); k = k + 1) {
      string path_logo = files[k];
      std::cout << path_logo << std::endl;
      if (path_logo.find("jpg") == string::npos)
        continue;
      Mat logo = imread(path_logo);   
      Mat logo_gray;
      if (logo.channels() == 3)
      cv::cvtColor(logo, logo_gray, CV_BGR2GRAY);  
      // imshow("logo", logo);
      Mat logo_harris = logo.clone();
      Mat logo_Shi_Tomasi = logo.clone();

      //-- create detector and descriptor --
      // if you want it faster, take e.g. FAST or ORB
      cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("HARRIS"); 

      // if you want it faster take e.g. ORB or FREAK
      cv::Ptr<cv::DescriptorExtractor> descriptor = cv::DescriptorExtractor::create("SURF"); 
      
      // detect keypoints
      std::vector<cv::KeyPoint> keypoints;
      detector->detect(logo_gray, keypoints);
      // extract features
      cv::Mat desc;
      descriptor->compute(logo_gray, keypoints, desc);
      std::cout << "Feature Dimension: " << desc.cols << " " << desc.rows;
      
      std::vector<cv::Point> keypoints_harris, keypoints_Shi_Tomasi;
      // detect harris corners
      cornerHarris_demo(logo_gray, logo_harris, keypoints_harris);
      // detect Shi_Tomasi corners
      goodFeaturesToTrack_Demo(logo_gray, logo_Shi_Tomasi, keypoints_Shi_Tomasi);
      // detect FAST keypoints
      FastFeatureDetector detector_FAST(100);
      vector<KeyPoint> keypoints_FAST;
      detector_FAST.detect(logo_gray, keypoints_FAST);
      

      Mat logo_keypoints;
      drawKeypoints(logo_gray, keypoints, logo_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
      imshow("harris", logo_harris);
      imshow("Shi_Tomasi", logo_Shi_Tomasi);
      imshow("KeyPoints", logo_keypoints);
      waitKey(0);

      // extract features from keypoints
      
    }
    */

    cv::Mat features_one_class;

    for (int k = 0; k < floor(0.8 * files.size()); k = k + 1) {
      string path_logo_a = files[k];
      if (k >= files.size() - 1)
        break;
      string path_logo_b = files[k + 1];

      // std::cout << path_logo_a << std::endl;
      // std::cout << path_logo_b << std::endl;

      Mat logo_a = imread(path_logo_a);
      Mat logo_b = imread(path_logo_b);
      if (!logo_a.data || !logo_b.data)
        continue;

      Mat logo_gray_a, logo_gray_b;
      if (logo_a.channels() == 3)
      cv::cvtColor(logo_a, logo_gray_a, CV_BGR2GRAY);  
      if (logo_b.channels() == 3)
      cv::cvtColor(logo_b, logo_gray_b, CV_BGR2GRAY);  

      //-- create detector and descriptor --
      // if you want it faster, take e.g. FAST or ORB
      cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("HARRIS"); 

      // if you want it faster take e.g. ORB or FREAK
      cv::Ptr<cv::DescriptorExtractor> descriptor = cv::DescriptorExtractor::create("SURF"); 
      
      // create descriptor matcher
      Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create("BruteForce"); 

      std::vector<cv::KeyPoint> keypoints_a, keypoints_b;
      detector->detect(logo_gray_a, keypoints_a);
      detector->detect(logo_gray_b, keypoints_b);

      // detect keypoints
      Mat logo_Shi_Tomasi = logo_a.clone();
      std::vector<cv::Point> keypoints_Shi_Tomasi;
      // detect Shi_Tomasi corners
      goodFeaturesToTrack_Demo(logo_gray_a, logo_Shi_Tomasi, keypoints_Shi_Tomasi);
      for (int k = 0; k < keypoints_Shi_Tomasi.size(); ++k) {
        KeyPoint kpt(keypoints_Shi_Tomasi[k].x, keypoints_Shi_Tomasi[k].y, 1, 0);
        keypoints_a.push_back(kpt);
      }

      // extract features
      cv::Mat desc_a, desc_b;
      descriptor->compute(logo_gray_a, keypoints_a, desc_a);
      descriptor->compute(logo_gray_b, keypoints_b, desc_b);

      if (features_one_class.cols == 0) {
        // std::cout << k << " " << desc_a.rows << " " << desc_a.cols << endl;
        features_one_class = desc_a.clone();
      }
      else {        
        // std::cout << k << " " << features_one_class.rows << " " << features_one_class.cols << endl;
        cv::vconcat(features_one_class, desc_a, features_one_class);
      }


      // std::cout << "Feature Dimension in A: " << desc_a.cols << " " << desc_a.rows << std::endl;
      // std::cout << "Feature Dimension in B: " << desc_b.cols << " " << desc_b.rows << std::endl;
      /*
      std::vector<cv::Point> keypoints_harris, keypoints_Shi_Tomasi;
      // detect harris corners
      cornerHarris_demo(logo_gray, logo_harris, keypoints_harris);
      // detect Shi_Tomasi corners
      goodFeaturesToTrack_Demo(logo_gray, logo_Shi_Tomasi, keypoints_Shi_Tomasi);
      // detect FAST keypoints
      FastFeatureDetector detector_FAST(100);
      vector<KeyPoint> keypoints_FAST;
      detector_FAST.detect(logo_gray, keypoints_FAST);
      */
      /*
      Mat logo_keypoints_a, logo_keypoints_b;
      if (keypoints_a.size() > 0)
      drawKeypoints(logo_gray_a, keypoints_a, logo_keypoints_a, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
      if (keypoints_b.size() > 0)
      drawKeypoints(logo_gray_b, keypoints_b, logo_keypoints_b, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
      // imshow("harris", logo_harris);
      // imshow("Shi_Tomasi", logo_Shi_Tomasi);
      imshow("KeyPoints_a", logo_keypoints_a);
      imshow("KeyPoints_b", logo_keypoints_b);
      */
      /*
      bool isWarpPerspective = 0;
      double ransacReprojThreshold = 3;
      RNG rng = theRNG();
      int mactherFilterType = getMatcherFilterType( "CrossCheckFilter");
      bool eval = false;
      doIteration( logo_a, logo_b, isWarpPerspective, keypoints_a, desc_a,
                 detector, descriptor, descriptor_matcher, mactherFilterType, eval,
                 ransacReprojThreshold, rng );
      */
      /*
      // match keypoints
      vector<DMatch> matches;
      // bool positive = matchingDescriptor(keypoints_a, keypoints_b, desc_a, desc_b, descriptor_matcher, matches, true);
      descriptor_matcher->match( desc_a, desc_b, matches );  
      Mat img_matches;  
      drawMatches(logo_a,keypoints_a,logo_b,keypoints_b,matches,img_matches,Scalar::all(-1),CV_RGB  (255,255,255),Mat(),4);  
      imshow("Matches_naive",img_matches);  
      waitKey(0);
      */
      /*
      vector<DMatch> matches;  
      descriptor_matcher->match( desc_a, desc_b, matches );  
  
      Mat img_matches;  
      drawMatches(logo_a,keypoints_a,logo_b,keypoints_b,matches,img_matches,Scalar::all(-1),CV_RGB  (255,255,255),Mat(),4);  
  
      imshow("Matches_naive",img_matches);  
      waitKey(0);
      */
    }
    cout << "Number of keypoints: " << features_one_class.rows << endl;
    // conduct kmeans for features of current class
    // 聚类3次，取结果最好的那次，聚类的初始化采用PP特定的随机算法
    std::cout << "Clustering..." << std::endl;
    Mat centers, labels;
    kmeans(features_one_class, clusterCount_local, labels,
               TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
               3, KMEANS_PP_CENTERS, centers);  
    std::cout << "Done" << std::endl;
    // count samples in each cluster and normalize
    Mat num_samples_cluster = cv::Mat(clusterCount_local, 1, CV_32FC1);
    for (int c = 0; c < clusterCount_local; ++c) {       
       num_samples_cluster.at<float>(c, 0) = 0;
    }
    for (int c = 0; c < labels.rows; ++c) {
       // cout << labels.at<int>(c) << " ";
       num_samples_cluster.at<float>(labels.at<int>(c), 0) += 1.0 / float(labels.rows);
    }
    for (int c = 0; c < clusterCount_local; ++c) {
       cout << num_samples_cluster.at<float>(c, 0) << " ";
    }
    cout << endl;
    cout << centers.cols << " " << centers.rows << endl;
    // combine centers and numbers into one matrix, and then save it
    Mat clusterInfo;
    hconcat(num_samples_cluster, centers, clusterInfo);
    string path_clusterInfo = folder + "/clusterInfo.xml";
    FileStorage fs(path_clusterInfo, FileStorage::WRITE);
    fs << "clusterInfo" << clusterInfo;    

    if (features_all_classes.cols == 0) {
      // std::cout << k << " " << desc_a.rows << " " << desc_a.cols << endl;
      features_all_classes = features_one_class.clone();
      for (int id = 0; id < features_one_class.rows; ++id)
        idx_all_classes.push_back(i);
    }
    else {        
      // std::cout << k << " " << features_one_class.rows << " " << features_one_class.cols << endl;
      cv::vconcat(features_all_classes, features_one_class, features_all_classes);
      for (int id = 0; id < features_one_class.rows; ++id)
        idx_all_classes.push_back(i);
    }
  }

  // 
  std::cout << "Clustering on all..." << std::endl;
  Mat centers_all_classes, labels_all_classes;
  kmeans(features_all_classes, clusterCount_global, labels_all_classes,
             TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
             3, KMEANS_PP_CENTERS, centers_all_classes);  
  std::cout << "Done" << std::endl;

  // count samples in each cluster for all classes
  Mat num_samples_cluster = cv::Mat(clusterCount_global, 1, CV_32FC1);
  for (int c = 0; c < clusterCount_global; ++c) {       
    num_samples_cluster.at<float>(c, 0) = 0;
  }
  for (int c = 0; c < labels_all_classes.rows; ++c) {
    // cout << labels.at<int>(c) << " ";
   num_samples_cluster.at<float>(labels_all_classes.at<int>(c), 0) += 1.0 / float(labels_all_classes.rows);
  }
  Mat clusterInfo_all_classes;
  hconcat(num_samples_cluster, centers_all_classes, clusterInfo_all_classes);
  string path_clusterInfo_global = imgfolder + "/clusterInfo_global.xml";
  FileStorage fs(path_clusterInfo_global, FileStorage::WRITE);
  fs << "clusterInfo_global" << clusterInfo_all_classes;    

  // count samples in each cluster for each class
  for (int i = 0; i < folders.size(); ++i) {

    Mat num_samples_cluster = cv::Mat(clusterCount_global, 1, CV_32FC1);
    for (int c = 0; c < clusterCount_global; ++c) {       
      num_samples_cluster.at<float>(c, 0) = 0;
    }
    for (int c = 0; c < labels_all_classes.rows; ++c) {
       if (idx_all_classes[c] == i)
       // cout << labels.at<int>(c) << " ";
       num_samples_cluster.at<float>(labels_all_classes.at<int>(c), 0) += 1.0;
    }
    string path_distribution_global = imgfolder + "/" + folders[i] + "/distribution_global.xml";
    FileStorage fs(path_distribution_global, FileStorage::WRITE);
    fs << "distribution" << num_samples_cluster;    
  }  
}
