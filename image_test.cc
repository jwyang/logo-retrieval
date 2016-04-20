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

int thresh = 100;
int max_thresh = 255;

int maxCorners = 50;
int maxTrackbar = 100;

RNG rng(12345);
int queryClass = 1;
string imgfolder = "../logo/tmplogo";

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

int main(int argc, char** argv) {

  std::vector<string> folders;
  fileutil::GetDir(imgfolder, &folders);
  std::cout << "Number of folders: " << folders.size() << std::endl;

  // create keypoint detector, descriptor
  //-- create detector and descriptor --
  // if you want it faster, take e.g. FAST or ORB
  cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("HARRIS"); 

  // if you want it faster take e.g. ORB or FREAK
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::DescriptorExtractor::create("SURF"); 
      
  // create descriptor matcher
  cv::Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create("BruteForce"); 

  // read global cluster info
  FileStorage fs(imgfolder + "/" + "clusterInfo_global.xml", FileStorage::READ);
  Mat clusterInfo_global;
  fs["clusterInfo_global"] >> clusterInfo_global;

  // read local clusters info 
  std::vector<cv::Mat> clusterInfos(folders.size());
  std::vector<cv::Mat> distribution_global(folders.size());
  for (int i = 0; i < folders.size(); ++i) {
    std::vector<string> files;
    string folder = imgfolder + "/" + folders[i]; 
    string path_clusterinfo = folder + "/" + "clusterInfo.xml";
    std::cout << path_clusterinfo << std::endl;
    fs.open(path_clusterinfo, FileStorage::READ);
    Mat clusterinfo;
    fs["clusterInfo"] >> clusterinfo;
    clusterInfos[i] = clusterinfo.clone();

    string path_distribution_global = folder + "/" + "distribution_global.xml";
    std::cout << path_distribution_global << std::endl;
    fs.open(path_distribution_global, FileStorage::READ);
    Mat distribution;
    fs["distribution"] >> distribution;
    distribution_global[i] = distribution.clone();
  }   
  std::cout << "Load information done." << std::endl;
  /*
  std::vector<cv::Mat> logos(folders.size());
  for (int i = 0; i < folders.size(); ++i) {
    std::vector<string> files;
    string folder = imgfolder + "/" + folders[i]; 
    fileutil::GetDirRecursive(folder, &files);
    for (int k = 0; k < 1; ++k) {
      string path_logo = files[k];
      std::cout << path_logo << std::endl;
      logos[i] = imread(path_logo);
    }
  }
  */

  std::cout << "Load templates done." << std::endl;
  for (int i = 0; i < folders.size(); ++i) {
    std::vector<string> files;
    string folder = imgfolder + "/" + folders[i]; 
    std::cout << folder << std::endl;
    fileutil::GetDirRecursive(folder, &files);
    std::cout << "Number of images: " << files.size() << std::endl;
    
    for (int j = ceil(0.8 * files.size()); j < files.size(); ++j) {
      string path_logo = files[j];
      std::cout << path_logo << std::endl;
      Mat logo = imread(path_logo);
      if (!logo.data)
        continue;

      Mat logo_gray, logo_gray_;
      cv::cvtColor(logo, logo_gray, CV_BGR2GRAY);

      // detect keypoints
      std::cout << "Detect keypoints..." << std::endl;
      std::vector<cv::KeyPoint> keypoints_probe;
      // detector->detect(logo_gray, keypoints_probe);

      Mat logo_Shi_Tomasi = logo.clone();
      std::vector<cv::Point> keypoints_Shi_Tomasi;
      // detect Shi_Tomasi corners
      goodFeaturesToTrack_Demo(logo_gray, logo_Shi_Tomasi, keypoints_Shi_Tomasi);
      for (int k = 0; k < keypoints_Shi_Tomasi.size(); ++k) {
        KeyPoint kpt(keypoints_Shi_Tomasi[k].x, keypoints_Shi_Tomasi[k].y, 1, 0);
        keypoints_probe.push_back(kpt);
      }
      if (keypoints_probe.size() == 0)
       continue;
      std::cout << "Detected " << keypoints_probe.size() << " points" << std::endl;
      
      Mat logo_keypoints;
      drawKeypoints(logo_gray, keypoints_probe, logo_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
      imshow("logo", logo_keypoints);
      waitKey(0);
      

      // detector->detect(logo_gray_b, keypoints_b);
      // extract features
      std::cout << "Extract features..." << std::endl;
      cv::Mat desc;
      SurfDescriptorExtractor extractor;
      // extractor.compute(logo_gray, keypoints_probe, desc);
      descriptor->compute(logo_gray, keypoints_probe, desc);

      // predict the brand of current logo
      std::cout << "Predict brand..." << std::endl;
      vector<float> scores(clusterInfos.size());     
      std::cout << clusterInfo_global.rows << " " << clusterInfo_global.cols << std::endl;
      Rect roi = Rect(1, 0, clusterInfo_global.cols - 1, clusterInfo_global.rows);
      Mat centers = clusterInfo_global(roi);
      Mat probs_global = clusterInfo_global.col(0);

      vector<float> probs_probe(centers.rows);
      for (int m = 0; m < desc.rows; ++m) {
         int idx_optimal;
         float dist_minimal(100);
         for (int c = 0; c < centers.rows; ++c) {
           if (norm(desc.row(m), centers.row(c)) < dist_minimal) {
             idx_optimal = c;
             dist_minimal = norm(desc.row(m), centers.row(c));
           }
         }
         // std::cout << dist_minimal << " ";
         // if (dist_minimal < 0.2)
         // scores[k] += probs_gallery[idx_optimal] * exp(-dist_minimal);
         probs_probe[idx_optimal] += 1.0 / float(desc.rows);
         
      }
      for (int h = 0; h < probs_probe.size(); ++h) {
        // std::cout << probs_probe[h] << " ";
        // scores[k] += fabs(probs_probe[h] - probs_gallery[h]) / (probs_probe[h] + probs_gallery[h]);
      }
      // std::cout << std::endl; 
     
      for (int k = 0; k < distribution_global.size(); ++k) {
        // std::cout << "Get probs of gallery" << std::endl;
        std::vector<float> probs_gallery(centers.rows);
        // std::cout << distribution_global[k].rows << " ";

        for (int h = 0; h < distribution_global[k].rows; ++h) {
          probs_gallery[h] = distribution_global[k].at<float>(h) / sum(distribution_global[k]).val[0];  
          // std::cout << probs_gallery[h] << " ";   
        }

        // std::cout << std::endl;

        scores[k] = 0;
        float score = 0;
        for (int h = 0; h < probs_probe.size(); ++h) {
          if (probs_global.at<float>(h) >= 1 / 50 )
          // std::cout << probs_probe[h] << " " << probs_gallery[h] << " ";
          score += probs_probe[h] * (log(probs_gallery[h] + 1e-5) - log(probs_global.at<float>(h) + 1e-5)); 
          // fabs(probs_probe[h] - probs_gallery[h]) / (probs_probe[h] + probs_gallery[h]);
        }
        // std::cout << std::endl;
        std::cout << score << " ";
        scores[k] = score;
        // std::cout << std::endl;      
      }
      std::cout << std::endl;
      std::cout << scores[i];
      std::cout << std::endl;

      // sort the scores and then print the first 5 matched brands
      std::sort(scores.begin(), scores.end());
      for (int k = 0; k < distribution_global.size(); ++k) {
        std::cout << scores[k] << " ";
      }
      std::cout << std::endl;
      /*
      for (int k = 0; k < clusterInfos.size(); ++k) {
        Rect roi = Rect(1, 0, clusterInfos[k].cols - 1, clusterInfos[k].rows);
        Mat centers = clusterInfos[k](roi);
        // std::cout << "Get probs of gallery" << std::endl;
        std::vector<float> probs_gallery(clusterInfos[k].rows);
        for (int c = 0; c < clusterInfos[k].rows; ++c) {
          probs_gallery[c] = clusterInfos[k].at<float>(c, 0);  
          // std::cout << probs_gallery[c] << " ";        
        }
        // std::cout << std::endl;
        scores[k] = 0;
        // std::cout << "Get probs of probe" << std::endl;
        vector<float> probs_probe(probs_gallery.size());
        for (int m = 0; m < desc.rows; ++m) {
           int idx_optimal;
           float dist_minimal(100);
           for (int c = 0; c < centers.rows; ++c) {
             if (norm(desc.row(m), centers.row(c)) < dist_minimal) {
               idx_optimal = c;
               dist_minimal = norm(desc.row(m), centers.row(c));
             }
           }
           // std::cout << dist_minimal << " ";
           // if (dist_minimal < 0.2)
           scores[k] += probs_gallery[idx_optimal] * exp(-dist_minimal);
           probs_probe[idx_optimal] += 1.0 / float(desc.rows);
        }
        std::cout << scores[k] / desc.rows << " ";
        /*
        for (int c = 0; c < clusterInfos[k].rows; ++c) {          
          std::cout << probs_probe[c] << " ";        
        }
        std::cout << std::endl;
        */

        // compute the distance between two probs
        /*
        scores[k] = 0;
        for (int h = 0; h < probs_probe.size(); ++h) {
          // std::cout << probs_probe[h] << " " << probs_gallery[h] << " ";
          scores[k] += fabs(probs_probe[h] - probs_gallery[h]) / (probs_probe[h] + probs_gallery[h]);
        }
        // std::cout << std::endl;
        std::cout << scores[k] << " ";
        */
      // }    
       

      // imshow("probe", logo);
      // waitKey(0);      
    }
  }
}
