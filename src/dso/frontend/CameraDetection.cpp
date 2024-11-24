//
// Created by zk on 24-4-8.
//
#include "CameraDetection.h"

#include <Eigen/Core>
#include <chrono>
#include <filesystem>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <thread>

#include "apriltags/Tag36h11.h"
#include "apriltags/TagDetection.h"
#include "apriltags/TagDetector.h"
#include "dotdetect/ImageProcessing.h"
#include "frontend_define.h"
//#include "../basic/macro_define.h"
#include "wlog.h"

namespace dso {

// oneCamJson

void CamCalib::CameraDetection::loadCameraData(bool skip_half_data) {
  for (const auto &[cam_id, cam_json_path] : m_config.camJsonPaths) {
    CalibIO::CameraJsonData camJsonRes;
    bool loadSuccess = CalibIO::camJsonLoad(cam_json_path, cam_id, camJsonRes);
    if (loadSuccess) {
      m_multiCamJsonRes[cam_id] = camJsonRes;
      YLOG_TRACE("cam_id is %d,cam_json_path is %s", cam_id,
                 cam_json_path.c_str());
    } else {
      LOG_FRONT_ERROR("Failed to load cam json. cam: zu%, path: %s\n", cam_id,
                      cam_json_path.c_str());
      std::exit(1);
    }
  }
  if (!camJsonAlign(m_multiCamJsonRes, m_config.frame_mini_gap)) {
    YLOG_ERROR("Camera align failed");
    exit(1);
  }
  LOG_FRONT_INFO("Camera frame aligned size: %d\n",
                 m_multiCamJsonRes.begin()->second.frames.size());

  if (skip_half_data) {
    for (const auto cid : m_config.calib_cid) {
      std::vector<CalibIO::FrameData> half_frames;
      for (int j = 0; j < m_multiCamJsonRes[cid].frames.size(); ++j) {
        if (j % 2 == 0) {
          half_frames.emplace_back(m_multiCamJsonRes[cid].frames[j]);
        }
      }
      m_multiCamJsonRes[cid].frames = half_frames;
    }
  }
}

void CamCalib::CameraDetection::loadImuData(CalibIO::ImuJsonData *p_imuData) {
  bool loadSuccess = CalibIO::imuJsonLoad(m_config.imuJsonPath, p_imuData);
  if (!loadSuccess) {
    LOG_FRONT_ERROR("Failed to load imu json. path: %s\n",
                    m_config.imuJsonPath.c_str());
    std::exit(-1);
  }
}

void CamCalib::CameraDetection::dotDetection() {
  const std::string &binFile = m_config.binFilePath;
  const std::unordered_map<int, std::string> &camPicPath = m_config.camPaths;

  for (const auto &[_, oneCam] : m_multiCamJsonRes) {
    int camId = oneCam.camId;
    const std::string &rootPath = camPicPath.at(camId);
    cv::Mat cvPic;
    DotDetect::ImageProcessing image_processing(
        m_config.dot_config.grid_spacing, m_config.dot_config.grid_size,
        m_config.dot_config.grid_seed, m_config.plate_num);
    image_processing.verbose = m_config.dot_config.show_detect_picture;

    DotDetect::ParamsImageProcessing curParams(25, 480);
    curParams.black_on_white = m_config.dot_config.black_dot;
    curParams.at_threshold = m_config.dot_config.adaptive_thresh;
    curParams.at_window_ratio = m_config.dot_config.window_ratio;
    curParams.conic_min_area = m_config.dot_config.conic_min_area;
    curParams.conic_symmetry = m_config.dot_config.conic_symmetry;
    curParams.conic_min_aspect = m_config.dot_config.conic_min_aspect;
    curParams.unique_size = m_config.dot_config.unique_size;

    CalibIO::CamFrames curCam;
    curCam.camID = camId;
    curCam.filePath = rootPath;
    int allCount = 0;
    for (const auto &frame : oneCam.frames) {
      std::string curImgPath = rootPath + "/" + frame.filename;
      cvPic = cv::imread(curImgPath, 0);
      if (cvPic.empty()) {
        LOG_FRONT_WARN("Pic can't load %s\n", curImgPath.c_str());
        continue;
      }
      int framePointsCount = 0;
      CalibIO::CurFrameRes curRes;
      image_processing.ProcessPic(cvPic, curParams, curRes);
      //      cv::Mat rgbshow;
      //      cv::cvtColor(cvPic, rgbshow, cv::COLOR_GRAY2RGB);
      curRes.frameName = frame.filename;
      curRes.timestamp = double(frame.timestamp) * 1e-9;
      curRes.exposure = double(frame.exposure_time) * 1e-9;
      curRes.gain = frame.gain;
      for (const auto &plateInfo : curRes.mImagePointSets) {
        curCam.obPlates.insert(plateInfo.first);

        for (const auto &picPoint : curRes.mImagePointSets) {
          framePointsCount += (int)picPoint.second.size();
        }
        //        cv::imshow("board:"+ to_string(plateInfo.first), rgbshow);
        //        cv::waitKey(0);
        //        cv::destroyAllWindows();
      }
      LOG_FRONT_INFO("detect: %s, points: %d\n", curImgPath.c_str(),
                     framePointsCount);
      allCount += framePointsCount;
      curCam.eachFrameInfo.emplace_back(curRes);
    }
    LOG_FRONT_INFO("one camera points num: %d\n", allCount);
    m_multiCamFrames.emplace(camId, curCam);
  }

  CalibIO::saveCamFrames(m_multiCamFrames, binFile);
  LOG_FRONT_INFO("Save Dot detect result into bin: %s\n", binFile.c_str());
}

void CamCalib::CameraDetection::dotSingleThreadDetect(
    const int &camId, const CalibIO::ConfigData &config,
    const DotDetect::ParamsImageProcessing &curParams,
    const DotDetect::ImageProcessing &image_processing,
    std::vector<CalibIO::FrameData *> &picJson,
    std::vector<CalibIO::CurFrameRes *> &picRes) {
  // std::cout << "camId is " << camId << std::endl;
  const std::string &rootPath = config.camPaths.at(camId);
  cv::Mat cvPic;

  bool binaryPic_init = false;
  cv::Mat binaryPic; // = cv::Mat::zeros(cvPic.rows, cvPic.cols, CV_8UC1);

  for (int frameID = 0; frameID < picJson.size(); ++frameID) {
    //    std::cout << "process frame " << frameID << std::endl;
    CalibIO::CurFrameRes *curRes = picRes[frameID];
    if (!config.skip_detection) {
      std::string curImgPath = rootPath + "/" + picJson[frameID]->filename;
      cvPic = cv::imread(curImgPath, 0);
      if (cvPic.empty()) {
        continue;
      }

      if (!binaryPic_init) {
        binaryPic = cv::Mat::zeros(cvPic.rows, cvPic.cols, CV_8UC1);
        binaryPic_init = true;
      }

      // CalibIO::CurFrameRes* curRes = picRes[frameID];

      image_processing.ProcessPic(cvPic, curParams, *curRes, &binaryPic);
    }
    binaryPic.setTo(0);
    curRes->frameName = picJson[frameID]->filename;
    curRes->timestamp = double(picJson[frameID]->timestamp) * 1e-9;
    curRes->exposure = double(picJson[frameID]->exposure_time) * 1e-9;
    curRes->gain = picJson[frameID]->gain;
    //    int points = 0;
    //    for (const std::pair<const int, std::vector<Eigen::Vector2d>>&
    //    boardPoints : curRes->mImagePointSets)
    //      points += boardPoints.second.size();
    //    std::printf("Frame %s detect points: %d \n",
    //    picJson[frameID]->filename.c_str(), points);
  }
}

/*
 * tagId
 *
 * a+rows a+rows+1
 * y
 * ^
 * |  3 - 2 -- 7 - 6
 * |  - a - -- -a+1-
 * |  0 - 1 -- 4 - 5
 * ------> x
 * */
int CamCalib::CameraDetection::computeAprilTagPatternID(
    int tagId, int pattern_num,
    const std::vector<std::pair<int, int>> &startID_endIDs) {
  for (int i = 0; i < pattern_num; i++) {
    if (tagId >= startID_endIDs[i].first && tagId <= startID_endIDs[i].second)
      return i;
  }
  std::cerr << " tag ID " << tagId << " is out of range" << std::endl;
  exit(1);
}

void CamCalib::CameraDetection::aprilTagThreadDetection(
    const int &camId, const CalibIO::ConfigData &config,
    const AprilTags::TagDetector &detector,
    std::vector<CalibIO::FrameData *> &picJson,
    std::vector<CalibIO::CurFrameRes *> &picRes,
    const std::vector<std::pair<int, int>> &startID_endIDs,
    const Eigen::Matrix<number_t, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor> &grid_points) {
  assert(picJson.size() == picRes.size());
  cv::Mat cvPic, cvPicColor;
  bool success = false;
  for (int i = 0; i < picJson.size(); ++i) {
    std::string curImgPath =
        config.camPaths.at(camId) + "/" + picJson[i]->filename;
    cvPic = cv::imread(curImgPath, 0);
    if (cvPic.empty()) {
      LOG_FRONT_WARN("Pic can't load: %s\n", curImgPath.c_str());
      continue;
    }

    CalibIO::CurFrameRes *curRes = picRes[i];
    std::vector<AprilTags::TagDetection> detections;
    cv::Mat tagCorners;
    std::vector<int> tag_per_board;
    if (!config.skip_detection) {
      detections = detector.extractTags(cvPic);

      if (config.apriltag_config.show_tag_detect)
        cv::cvtColor(cvPic, cvPicColor, cv::COLOR_GRAY2RGB);
      std::vector<AprilTags::TagDetection>::iterator iter = detections.begin();
      for (iter = detections.begin(); iter != detections.end();) {
        bool remove = false;
        for (int j = 0; j < 4; j++) {
          remove |=
              iter->p[j].first < config.apriltag_config.min_border_distance;
          remove |=
              iter->p[j].first >
              (float)(cvPic.cols) - config.apriltag_config.min_border_distance;
          remove |=
              iter->p[j].second < config.apriltag_config.min_border_distance;
          remove |=
              iter->p[j].second >
              (float)(cvPic.rows) - config.apriltag_config.min_border_distance;
        }
        if (iter->good != 1)
          remove |= true;
        if (iter->id >= (int)config.apriltag_config.one_board_tags *
                                config.apriltag_config.pattern_num +
                            config.apriltag_config.board_offset)
          remove |= true;
        if (remove) {
          iter = detections.erase(iter);
        } else {
          ++iter;
        }
      }
      // std::vector<int> tag_per_board;
      success = false;
      tag_per_board.resize(config.apriltag_config.pattern_num, 0);
      for (const auto &detection : detections) {
        int patternID = computeAprilTagPatternID(detection.id, config.plate_num,
                                                 startID_endIDs);
        tag_per_board[patternID]++;
      }
      for (int pattern_num = 0;
           pattern_num < config.apriltag_config.pattern_num; pattern_num++) {
        if (tag_per_board[pattern_num] >=
            config.apriltag_config.min_tags_for_valid_obs)
          success = true;
      }
      std::sort(detections.begin(), detections.end(),
                AprilTags::TagDetection::sortByIdCompare);
      if (detections.size() > 1) {
        for (int i = 0; i < detections.size() - 1; i++) {
          if (detections[i].id == detections[i + 1].id) {
            success = false; //  duplicate apriltags are detected, if this
                             //  happens, skip this frame;
            break;
          }
        }
      }
      tagCorners = cv::Mat(4 * detections.size(), 2, CV_32F);

      for (unsigned i = 0; i < detections.size(); i++) {
        for (unsigned j = 0; j < 4; j++) {
          tagCorners.at<float>(4 * i + j, 0) = detections[i].p[j].first;
          tagCorners.at<float>(4 * i + j, 1) = detections[i].p[j].second;
        }
      }
    } else {
      success = false;
    }
    if (!success) {
      curRes->frameName = picJson[i]->filename;
      curRes->timestamp = double(picJson[i]->timestamp) * 1e-9;
      curRes->exposure = double(picJson[i]->exposure_time) * 1e-9;
      curRes->gain = picJson[i]->gain;
      picRes[i] = curRes;
      continue;
    }
    cv::Mat tagCornersRaw = tagCorners.clone();
    if (config.apriltag_config.do_sub_pixel_refine) {
#ifdef USE_HOMO_WARP
      int iter_max = 2;
      int iter_count = 0;
      int count = 0;
      int image_res = 70; // 50;  // 70;
      int image_res_padd =
          static_cast<int>(0.2 * static_cast<double>(image_res));
      int sub_pix_window_size = 5;

      if (config.is_rgb) {
        image_res = 50;
        image_res_padd =
            static_cast<int>(0.15 * static_cast<double>(image_res));
        sub_pix_window_size = 5;
      }

      cv::Mat xMat(image_res, image_res, CV_32F);
      cv::Mat yMat(image_res, image_res, CV_32F);
      // Eigen::Matrix<double, 3, image_res * image_res> control_mat,
      // target_mat, temp_mat;
      Eigen::MatrixXd control_mat, target_mat, temp_mat;
      control_mat.resize(3, image_res * image_res);
      target_mat.resize(3, image_res * image_res);
      temp_mat.resize(3, image_res * image_res);

      Eigen::Matrix<double, 3, 4> control_mat_corner, temp_mat_corner,
          target_mat_corner, source_mat, control_mat2, control_mat3,
          source_mat2;
      control_mat.row(2).setOnes();
      control_mat_corner.row(2).setOnes();
      source_mat.row(2).setOnes();
      control_mat3.row(2).setOnes();
      std::vector<std::set<int>> hori_set, vert_set;
      std::vector<cv::Point2f> control_points = {
          cv::Point2f(0, image_res), cv::Point2f(image_res, image_res),
          cv::Point2f(image_res, 0), cv::Point2f(0, 0)};
      std::vector<cv::Point2f> control_points_pad = {
          cv::Point2f(-image_res_padd, image_res + image_res_padd),
          cv::Point2f(image_res + image_res_padd, image_res + image_res_padd),
          cv::Point2f(image_res + image_res_padd, -image_res_padd),
          cv::Point2f(-image_res_padd, -image_res_padd)};

      cv::Mat tagCorners_homo(4, 2, CV_32F);
      for (int row = 0; row < image_res; row++) {
        for (int col = 0; col < image_res; col++) {
          xMat.at<float>(row, col) = float(col);
          yMat.at<float>(row, col) = float(row);
          control_mat.block(0, count, 2, 1) << double(col), double(row);
          count++;
        }
      }
      Eigen::Matrix3d Homo;
      // std::cout << "control_mat:\n" << control_mat << std::endl;
      for (int i = 0; i < detections.size(); i++) {
#ifdef DO_2ND_ROUND_CORNER_OPT
        iter_count = 0;
        while (iter_count < iter_max) {
          // tagCornersRaw = tagCorners.clone();
#endif
          std::vector<cv::Point2f> source_points1, source_points2;
          bool draw = false;
          for (int j = 0; j < 4; j++) {
            source_points1.emplace_back(
                cv::Point2f(tagCorners.at<float>(4 * i + j, 0),
                            tagCorners.at<float>(4 * i + j, 1)));
            source_mat.block(0, j, 2, 1) =
                Eigen::Vector2d(tagCorners.at<float>(4 * i + j, 0),
                                tagCorners.at<float>(4 * i + j, 1));
            //              std::cout << "source points1: " << source_points1[j]
            //              << std::endl;
          }
          cv::Mat h = cv::findHomography(control_points, source_points1);
          //          std::cout << "h1:\n" << h << std::endl;
          cv::cv2eigen(h, Homo);
          control_mat_corner.block(0, 0, 2, 1) =
              Eigen::Vector2d(control_points_pad[0].x, control_points_pad[0].y);
          control_mat_corner.block(0, 1, 2, 1) =
              Eigen::Vector2d(control_points_pad[1].x, control_points_pad[1].y);
          control_mat_corner.block(0, 2, 2, 1) =
              Eigen::Vector2d(control_points_pad[2].x, control_points_pad[2].y);
          control_mat_corner.block(0, 3, 2, 1) =
              Eigen::Vector2d(control_points_pad[3].x, control_points_pad[3].y);
          target_mat_corner = Homo * control_mat_corner;
          temp_mat_corner.row(0) = target_mat_corner.row(2);
          temp_mat_corner.row(1) = target_mat_corner.row(2);
          temp_mat_corner.row(2) = target_mat_corner.row(2);
          target_mat_corner =
              target_mat_corner.array() / temp_mat_corner.array();

          for (int j = 0; j < 4; j++) {
            source_points2.emplace_back(
                cv::Point2f(target_mat_corner(0, j), target_mat_corner(1, j)));
            //              std::cout << "source points2: " << source_points2[j]
            //              << std::endl;
          }
          h = cv::findHomography(control_points, source_points2);
          //          std::cout << "h2:\n" << h << std::endl;
          cv::cv2eigen(h, Homo);
          target_mat = Homo * control_mat;
          temp_mat.row(0) = target_mat.row(2);
          temp_mat.row(1) = target_mat.row(2);
          temp_mat.row(2) = target_mat.row(2);
          target_mat = target_mat.array() / temp_mat.array();

          control_mat2 = Homo.inverse() * source_mat;
          temp_mat_corner.row(0) = control_mat2.row(2);
          temp_mat_corner.row(1) = control_mat2.row(2);
          temp_mat_corner.row(2) = control_mat2.row(2);
          control_mat2 = control_mat2.array() / temp_mat_corner.array();

          //          std::cout << "target_mat:\n" << target_mat << std::endl;
          count = 0;
          for (int row = 0; row < image_res; row++) {
            for (int col = 0; col < image_res; col++) {
              xMat.at<float>(row, col) = target_mat(0, count);
              yMat.at<float>(row, col) = target_mat(1, count);
              count++;
            }
          }
          //          std::cout << "xMat: \n" << xMat << std::endl;
          //          std::cout << "yMat: \n" << yMat << std::endl;
          cv::Mat targetImage, image_temp1;
          if (config.apriltag_config.show_tag_detect) {
            image_temp1 = cvPic.clone();
            cv::cvtColor(image_temp1, image_temp1, cv::COLOR_GRAY2BGR);
          }
          cv::remap(cvPic, targetImage, xMat, yMat, cv::INTER_CUBIC);

          for (int j = 0; j < 4; j++) {
            tagCorners_homo.at<float>(j, 0) = control_mat2(0, j);
            tagCorners_homo.at<float>(j, 1) = control_mat2(1, j);
          }
          cv::cornerSubPix(
              targetImage, tagCorners_homo,
              cv::Size(sub_pix_window_size, sub_pix_window_size),
              cv::Size(-1, -1), // 2,2
              cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.05));

          for (int j = 0; j < 4; j++) {
            control_mat3(0, j) = tagCorners_homo.at<float>(j, 0);
            control_mat3(1, j) = tagCorners_homo.at<float>(j, 1);
          }

          source_mat2 = Homo * control_mat3;
          temp_mat_corner.row(0) = source_mat2.row(2);
          temp_mat_corner.row(1) = source_mat2.row(2);
          temp_mat_corner.row(2) = source_mat2.row(2);
          source_mat2.array() = source_mat2.array() / temp_mat_corner.array();

          for (int j = 0; j < 4; j++) {
            tagCorners.at<float>(4 * i + j, 0) = source_mat2(0, j);
            tagCorners.at<float>(4 * i + j, 1) = source_mat2(1, j);
            if (!draw) {
              if (tagCorners.at<float>(4 * i + j, 0) < 0 ||
                  tagCorners.at<float>(4 * i + j, 1) < 0 ||
                  tagCorners.at<float>(4 * i + j, 0) > cvPic.cols ||
                  tagCorners.at<float>(4 * i + j, 1) > cvPic.rows) {
                draw = true;
              }
            }
          }
          if (config.apriltag_config.show_tag_detect) {
            cv::cvtColor(targetImage, targetImage, cv::COLOR_GRAY2BGR);
            for (int jj = 0; jj < 4; jj++) {
              cv::circle(image_temp1, source_points2[jj], 3, CV_RGB(0, 0, 255),
                         cv::FILLED);
              cv::circle(image_temp1, source_points1[jj], 2, CV_RGB(255, 0, 0),
                         cv::FILLED);
              cv::circle(image_temp1,
                         cv::Point2f(tagCorners.at<float>(4 * i + jj, 0),
                                     tagCorners.at<float>(4 * i + jj, 1)),
                         1, CV_RGB(0, 255, 0), cv::FILLED);
              cv::circle(targetImage,
                         cv::Point2f(control_mat2(0, jj), control_mat2(1, jj)),
                         3, CV_RGB(0, 0, 255), cv::FILLED);
              cv::circle(targetImage,
                         cv::Point2f(tagCorners_homo.at<float>(jj, 0),
                                     tagCorners_homo.at<float>(jj, 1)),
                         2, CV_RGB(0, 255, 0), cv::FILLED);
            }
            if (/*true ||*/ draw || true) {
              std::cout << "cur_iter: " << iter_count << std::endl;
              cv::imshow("orig image", image_temp1);
              cv::imshow("warped image", targetImage);
              cv::imwrite("/home/roger/work/smartgit/calib001/yvrcalibration/"
                          "build/a.png",
                          image_temp1);
              cv::imwrite("/home/roger/work/smartgit/calib001/yvrcalibration/"
                          "build/b.png",
                          targetImage);
              cv::waitKey(0);
            }
          }
#ifdef DO_2ND_ROUND_CORNER_OPT
          iter_count++;
        }
#endif
      }
#else
      int ratio = std::max(cvPic.rows / 640, 1);
      cv::cornerSubPix(
          cvPic, tagCorners, cv::Size(2 * ratio, 2 * ratio),
          cv::Size(-1, -1), // 2,2
          cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30,
                           0.1));
#endif
    }
    std::vector<std::vector<bool>> outCornerObserved;
    Eigen::MatrixXd outImagePoints;
    outCornerObserved.resize(
        config.apriltag_config.pattern_num,
        std::vector<bool>(4 * config.apriltag_config.one_board_tags, false));
    outImagePoints.resize(4 * config.apriltag_config.one_board_tags,
                          2 * config.apriltag_config.pattern_num);
    std::map<int, std::vector<int>> gridIdMap;
    for (int i = 0; i < detections.size(); i++) {
      // get the tag id
      int tagId = detections[i].id;

      // calculate the grid idx for all four tag corners given the tagId and
      // cols
      int baseId = (int)(tagId / (config.apriltag_config.cols)) *
                       config.apriltag_config.cols * 4 +
                   (tagId % (config.apriltag_config.cols)) * 2;
      int pIdx[] = {baseId, baseId + 1,
                    baseId + (int)(2 * config.apriltag_config.cols) + 1,
                    baseId + (int)(2 * config.apriltag_config.cols)};

      // add four points per tag
      for (int j = 0; j < 4; j++) {
        // refined corners
        double corner_x = tagCorners.row(4 * i + j).at<float>(0);
        double corner_y = tagCorners.row(4 * i + j).at<float>(1);

        // raw corners
        double cornerRaw_x = tagCornersRaw.row(4 * i + j).at<float>(0);
        double cornerRaw_y = tagCornersRaw.row(4 * i + j).at<float>(1);

        // only add point if the displacement in the subpixel refinement is
        // below a given threshold
        double subpix_displacement_squarred =
            (corner_x - cornerRaw_x) * (corner_x - cornerRaw_x) +
            (corner_y - cornerRaw_y) * (corner_y - cornerRaw_y);

        // add all points, but only set active if the point has not moved to far
        // in the subpix refinement
        ///  image point order
        ///      pattern0_id 0    |    pattern1_id 0    |  ...
        ///      pattern0_id 1    |    pattern1_id 1    |
        ///      pattern0_id 2    |    pattern1_id 2    |
        ///      pattern0_id 3    |    pattern1_id 3    |
        ///      pattern0_id 4    |    pattern1_id 4    |
        ///      pattern0_id 5    |    pattern1_id 5    |
        ///             .                   .
        ///             .                   .
        int patternID =
            computeAprilTagPatternID(tagId, config.plate_num, startID_endIDs);
        int row_position = pIdx[j] - startID_endIDs[patternID].first * 4;
        int col_position = patternID * 2;
        if (j == 0) {
          // gridIdMap.insert(std::make_pair());
        }

        outImagePoints.block<1, 2>(row_position, col_position) =
            Eigen::Matrix<number_t, 1, 2>(corner_x, corner_y);

        if (subpix_displacement_squarred <=
            config.apriltag_config.max_subpix_displacement2 * 5) {
          outCornerObserved[patternID][row_position] = true;
        } else {
          outCornerObserved[patternID][row_position] = false;
        }
      }
    }

    for (int pattern_id = 0; pattern_id < config.apriltag_config.pattern_num;
         pattern_id++) {
      if (tag_per_board[pattern_id] >
          config.apriltag_config.min_points_obsetved) {
        if (curRes->mGridId.count(pattern_id) == 0) {
          curRes->mGridId.emplace(pattern_id, std::vector<int>{});
          curRes->mImagePointSets.emplace(pattern_id,
                                          std::vector<Eigen::Vector2d>{});
          curRes->mObjectPointSets.emplace(pattern_id,
                                           std::vector<Eigen::Vector3d>{});
        }

        for (int point_id = 0;
             point_id < 4 * config.apriltag_config.one_board_tags; point_id++) {
          if (outCornerObserved[pattern_id][point_id]) {
            curRes->mGridId.at(pattern_id)
                .emplace_back(4 * config.apriltag_config.one_board_tags *
                                  pattern_id +
                              point_id);
            curRes->mObjectPointSets.at(pattern_id)
                .emplace_back(grid_points.row(point_id).transpose());
            curRes->mImagePointSets.at(pattern_id)
                .emplace_back(
                    Vec2(outImagePoints(point_id, pattern_id * 2),
                         outImagePoints(point_id, pattern_id * 2 + 1)));
            if (config.apriltag_config.show_tag_detect) {
              cv::circle(
                  cvPicColor,
                  cv::Point2f(outImagePoints(point_id, pattern_id * 2),
                              outImagePoints(point_id, pattern_id * 2 + 1)),
                  2, CV_RGB(255, 0, 0), 2, cv::FILLED);
              cv::putText(
                  cvPicColor,
                  to_string(4 * config.apriltag_config.one_board_tags *
                                pattern_id +
                            point_id),
                  cv::Point2f(outImagePoints(point_id, pattern_id * 2) + 2,
                              outImagePoints(point_id, pattern_id * 2 + 1) - 2),
                  cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(255, 0, 255), 1);
            }
          }
        }
      }
    }
    if (config.apriltag_config.show_tag_detect) {
      cv::imshow("Aprilgrid: Tag detection", cvPicColor); // OpenCV call
      cv::waitKey(0);
    }

    curRes->frameName = picJson[i]->filename;
    curRes->timestamp = double(picJson[i]->timestamp) * 1e-9;
    curRes->exposure = double(picJson[i]->exposure_time) * 1e-9;
    curRes->gain = picJson[i]->gain;
    picRes[i] = curRes;
  }
}

void CamCalib::CameraDetection::multiThreadDetect() {
  for (auto &[_, oneCam] : m_multiCamJsonRes) {
    int camId = oneCam.camId;
    CalibIO::CamFrames curCam;
    curCam.eachFrameInfo.resize(oneCam.frames.size());
    int threadNum = m_config.detect_thread_num;
    std::vector<std::thread> multiThread;
    std::vector<std::vector<CalibIO::FrameData *>> multiDataJson(
        threadNum, std::vector<CalibIO::FrameData *>{});
    std::vector<std::vector<CalibIO::CurFrameRes *>> multiFrameRes(
        threadNum, std::vector<CalibIO::CurFrameRes *>{});
    std::printf("Multi thread detect: camera_%d\n", camId);

    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();

    for (int frameId = 0; frameId < oneCam.frames.size(); ++frameId) {
      int threadId = frameId % threadNum;
      multiDataJson[threadId].emplace_back(&oneCam.frames[frameId]);
      multiFrameRes[threadId].emplace_back(&curCam.eachFrameInfo[frameId]);
    }
    AprilTags::TagDetector detector(AprilTags::tagCodes36h11,
                                    m_config.apriltag_config.black_board);

    DotDetect::ImageProcessing image_processing(
        m_config.dot_config.grid_spacing, m_config.dot_config.grid_size,
        m_config.dot_config.grid_seed, m_config.plate_num);

    if (m_config.board_type == CalibIO::DOT) {
      image_processing.verbose = m_config.dot_config.show_detect_picture;
      DotDetect::ParamsImageProcessing curParams(
          m_config.dot_config.min_area_point_num, oneCam.cameraInfo.width);
      curParams.black_on_white = m_config.dot_config.black_dot;
      curParams.at_threshold = m_config.dot_config.adaptive_thresh;
      curParams.at_window_ratio = m_config.dot_config.window_ratio;
      curParams.conic_min_area = m_config.dot_config.conic_min_area;
      curParams.conic_symmetry = m_config.dot_config.conic_symmetry;
      curParams.conic_min_aspect = m_config.dot_config.conic_min_aspect;
      curParams.unique_size = m_config.dot_config.unique_size;
      curParams.skip_detection = m_config.skip_detection;
      if (threadNum > 1) {
        for (int i = 0; i < threadNum; ++i) {
          multiThread.emplace_back(
              dotSingleThreadDetect, camId, m_config, std::ref(curParams),
              std::ref(image_processing), std::ref(multiDataJson[i]),
              std::ref(multiFrameRes[i]));
          std::printf("thread_%d frames -> %zu\n", i, multiDataJson[i].size());
        }
      } else {
        dotSingleThreadDetect(camId, m_config, curParams, image_processing,
                              multiDataJson[0], multiFrameRes[0]);
      }
    } else if (m_config.board_type == CalibIO::APRILTAG) {
      if (threadNum > 1) {
        for (int i = 0; i < threadNum; ++i) {
          multiThread.emplace_back(
              aprilTagThreadDetection, camId, m_config, std::ref(detector),
              std::ref(multiDataJson[i]), std::ref(multiFrameRes[i]),
              m_startID_endIDs, m_grid_points);
          std::printf("thread_%d frames -> %zu\n", i, multiDataJson[i].size());
        }
      } else {
        aprilTagThreadDetection(camId, m_config, std::ref(detector),
                                multiDataJson[0], multiFrameRes[0],
                                m_startID_endIDs, m_grid_points);
      }
    } else {
      std::cerr << "not support board type!!" << std::endl;
      std::exit(-1);
    }

    for (auto &thread : multiThread) {
      thread.join();
    }
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::chrono::duration<double> duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::printf("detect time: %f s \n", duration.count());
    m_multiCamFrames.emplace(camId, curCam);
  }

  const std::string &binFile = m_config.binFilePath;
  CalibIO::saveCamFrames(m_multiCamFrames, binFile);
  LOG_FRONT_INFO("Save detect result into bin: %s\n", binFile.c_str());
}

bool CamCalib::CameraDetection::hasResBin() {
  if (m_config.calib_stage != CalibIO::GRAY_RGB) {
    std::ifstream file(m_config.binFilePath);
    if (file.is_open()) {
      file.close();
      return true;
    } else {
      LOG_FRONT_ERROR("Bin File does not exist: %s.\n",
                      m_config.binFilePath.c_str());
    }
    return false;
  } else {
    std::ifstream file_gray(m_config.binFilePath_gray);
    std::ifstream file_rgb(m_config.binFilePath_rgb);
    if (file_gray.is_open() && file_rgb.is_open()) {
      file_gray.close();
      file_rgb.close();
      return true;
    } else {
      LOG_FRONT_ERROR("gray or rgb Bin File does not exist: %s, %s\n",
                      m_config.binFilePath_gray.c_str(),
                      m_config.binFilePath_rgb.c_str());
    }
    return false;
  }
}

bool CamCalib::CameraDetection::loadBinFile() {
  LOG_FRONT_INFO("LoadBinFile: %s\n", m_config.binFilePath.c_str());
  if (m_config.calib_stage != CalibIO::GRAY_RGB) {
    CalibIO::loadCamFrames(m_config.binFilePath, m_multiCamFrames);
  } else {
    LOG_FRONT_INFO("LoadBinFile_gray: %s\n", m_config.binFilePath_gray.c_str());
    LOG_FRONT_INFO("LoadBinFile_rgb: %s\n", m_config.binFilePath_rgb.c_str());
    CalibIO::loadCamFrames(m_config.binFilePath_gray, m_multiCamFrames);
    CalibIO::loadCamFrames(m_config.binFilePath_rgb, m_multiCamFrames_rgb);
  }
  return false;
}

void CamCalib::CameraDetection::showDetection() {
  std::map<int, int> cid_to_index;
  int count = 0;
  for (const int &id : m_config.calib_cid) {
    cid_to_index.emplace(std::make_pair(id, count));
    count++;
  }
  for (const auto &oneCam : m_multiCamFrames) {
    // int camId = cid_to_index[oneCam.first];
    const std::string &camPath = m_config.camPaths.at(oneCam.first);
    for (const auto &curRes : oneCam.second.eachFrameInfo) {
      LOG_FRONT_INFO("load file: %s\n",
                     (camPath + "/" + curRes.frameName).c_str());
      cv::Mat curPic = cv::imread(camPath + "/" + curRes.frameName, 1);
      for (const auto &oneBoard : curRes.mGridId) {
        int boardId = oneBoard.first;
        for (int idx = 0; idx < curRes.mGridId.at(boardId).size(); ++idx) {
          cv::circle(curPic,
                     cv::Point(curRes.mImagePointSets.at(boardId)[idx].x(),
                               curRes.mImagePointSets.at(boardId)[idx].y()),
                     2,
                     cv::Scalar(boardId * 100, 255 - boardId * 100,
                                255 - 10 * boardId));
          //          cv::putText(back_pic,
          //          std::to_string(curRes.mGridId.at(boardId)[idx]),
          //                      cv::Point(curRes.mImagePointSets.at(boardId)[idx].x(),
          //                      curRes.mImagePointSets.at(boardId)[idx].y()),
          //                      1, 1, cv::Scalar(0, 255, 0));
        }
      }
      cv::imshow("show detect:" + curRes.frameName, curPic);
      cv::waitKey(0);
      cv::destroyAllWindows();
    }
  }
}

void CamCalib::CameraDetection::transRes(
    dso::aligned_vector<dso::CalibFrame> &res, bool is_rgb) {
  int pointNum = 0;
  // std::unordered_map<int/*camId*/, CalibIO::CamFrames> m_multiCamFrames;
  int camNum = (int)m_multiCamFrames.size();
  LOG_FRONT_INFO("camNum: %d\n", camNum);
  int frameNum = (int)m_multiCamFrames.begin()->second.eachFrameInfo.size();
  res.resize(frameNum, CalibFrame());
  //  std::map<int, int> cid_to_index;
  //  if (!is_rgb) {
  //    int count = 0;
  //    for (const int& id : m_config.calib_cid) {
  //      cid_to_index.emplace(std::make_pair(id, count));
  //      count++;
  //    }
  //  } else {
  //    int count = 0;
  //    for (const int& id : m_config.rgb_cid) {
  //      cid_to_index.emplace(std::make_pair(id, count));
  //      count++;
  //    }
  //  }
  for (const auto &oneCam : m_multiCamFrames) {
    int curCamId = oneCam.first;
    std::cout << "curCamId: " << curCamId << std::endl;
    const std::string &curCamPath = m_config.camPaths.at(curCamId);

    for (int frameId = 0; frameId < frameNum; ++frameId) {
      res[frameId].timestamp = oneCam.second.eachFrameInfo[frameId].timestamp;
      res[frameId].cid_to_gain[curCamId] =
          oneCam.second.eachFrameInfo[frameId].gain;
      res[frameId].cid_to_exposure_time[curCamId] =
          oneCam.second.eachFrameInfo[frameId].exposure;
      res[frameId].cid_to_img_file_path[curCamId] =
          curCamPath + "/" + oneCam.second.eachFrameInfo[frameId].frameName;
      res[frameId].cid_pid_to_point_vm[curCamId] = {}; // create empty
      for (const auto &oneBoard :
           oneCam.second.eachFrameInfo[frameId].mGridId) {
        int curBoardId = oneBoard.first;
        for (int idx = 0; idx < oneBoard.second.size(); ++idx) {
          const Eigen::Vector2d &curP2d =
              oneCam.second.eachFrameInfo[frameId].mImagePointSets.at(
                  curBoardId)[idx];
          const Eigen::Vector3d &curP3d =
              oneCam.second.eachFrameInfo[frameId].mObjectPointSets.at(
                  curBoardId)[idx];
          int curPid =
              oneCam.second.eachFrameInfo[frameId].mGridId.at(curBoardId)[idx];
          pointNum++;
          dso::PointVM pVm;
          pVm.xyz = curP3d;
          pVm.uv = curP2d;
          // std::cout << "pid: " << curPid << ", uv: " << curP2d.transpose() <<
          // std::endl;
          res[frameId].cid_pid_to_point_vm[curCamId][curPid] = pVm;
        }
      }
    }
  }
  LOG_FRONT_INFO("Point num: d%", pointNum);
}

void CamCalib::CameraDetection::dotDetect(const string &imgFolder,
                                          const std::string &saveFolder) {
  std::vector<std::string> filenames;
  GetBmpNames(imgFolder, filenames);
  DotDetect::ImageProcessing image_processing(
      m_config.dot_config.grid_spacing, m_config.dot_config.grid_size,
      m_config.dot_config.grid_seed, m_config.plate_num);
  image_processing.verbose = m_config.dot_config.show_detect_picture;

  DotDetect::ParamsImageProcessing curParams(
      m_config.dot_config.min_area_point_num, 480);
  curParams.black_on_white = m_config.dot_config.black_dot;
  curParams.at_threshold = m_config.dot_config.adaptive_thresh;
  curParams.at_window_ratio = m_config.dot_config.window_ratio;
  curParams.conic_min_area = m_config.dot_config.conic_min_area;
  curParams.conic_symmetry = m_config.dot_config.conic_symmetry;
  curParams.conic_min_aspect = m_config.dot_config.conic_min_aspect;
  curParams.unique_size = m_config.dot_config.unique_size;

  CalibIO::CamFrames curCam;
  cv::Mat cvPic;
  for (const auto &filename : filenames) {
    std::string curImgPath = imgFolder + "/" + filename;
    std::cerr << "detect img: " << curImgPath << std::endl;
    cvPic = cv::imread(curImgPath, 0);
    if (cvPic.empty()) {
      LOG_FRONT_WARN("Pic can't load %s\n", curImgPath.c_str());
      continue;
    }
    int framePointsCount = 0;
    CalibIO::CurFrameRes curRes;
    image_processing.ProcessPic(cvPic, curParams, curRes);

    cv::Mat showId;
    cv::cvtColor(cvPic, showId, cv::COLOR_GRAY2RGB);

    for (const auto &oneData : curRes.mImagePointSets) {
      framePointsCount += oneData.second.size();
      int boardId = oneData.first;
      for (int i = 0; i < oneData.second.size(); ++i) {
        int curPointId = curRes.mGridId.at(boardId)[i];
        cv::circle(showId,
                   cv::Point(oneData.second[i].x(), oneData.second[i].y()), 3,
                   cv::Scalar(0, 0, 255));
        if (curPointId % 5 == 0) {
          cv::putText(showId, std::to_string(curPointId),
                      cv::Point(oneData.second[i].x(), oneData.second[i].y()),
                      1, 1, cv::Scalar(255, 0, 0));
        }
      }
    }
    cv::imwrite(saveFolder + "/" + filename, showId);
    //    cv::waitKey(0);
    //    cv::destroyWindow(filename);

    LOG_FRONT_INFO("detect: %s, points: %d\n", curImgPath.c_str(),
                   framePointsCount);
  }
}

bool CamCalib::CameraDetection::FileEndsWith(const std::string &str,
                                             const std::string &suffix) {
  if (str.length() >= suffix.length()) {
    return (0 == str.compare(str.length() - suffix.length(), suffix.length(),
                             suffix));
  } else {
    return false;
  }
}

void CamCalib::CameraDetection::GetBmpNames(
    const std::string &path, std::vector<std::string> &filenames) {
  DIR *pDir;
  struct dirent *ptr;
  if (!(pDir = opendir(path.c_str()))) {
    std::cout << "Folder doesn't Exist!" << std::endl;
    return;
  }
  while ((ptr = readdir(pDir)) != nullptr) {
    if (FileEndsWith(ptr->d_name, ".bmp")) {
      filenames.emplace_back(ptr->d_name);
    }
  }
  std::sort(filenames.begin(), filenames.end());
  closedir(pDir);
}

void CamCalib::CameraDetection::drawPointsRangePic() const {
  //  m_multiCamFrames
  for (const auto &[_, oneCamData] : m_multiCamJsonRes) {
    int curCamId = oneCamData.camId;
    if (m_multiCamFrames.count(curCamId) == 0) {
      LOG_FRONT_ERROR("Camera %d did't get enough result", curCamId);
      std::exit(-1);
    }
    int gridPixels = 1, color_num = 5;
    int gridWidth = oneCamData.cameraInfo.width / gridPixels;
    int gridHeight = oneCamData.cameraInfo.height / gridPixels;

    int imageW = gridWidth + cvRound(static_cast<float>(gridWidth) *
                                     (16.f / 9.f - 4.f / 3.f));
    cv::Mat image_gray(gridHeight, imageW, CV_8UC1, cv::Scalar(0));

    std::string img_path = m_config.resultPath + "/Camera_" +
                           std::to_string(curCamId) + "_point_range.bmp";

    unsigned int observedPointsNum = 0;
    unsigned int coveragePointsNum = 0;

    for (const CalibIO::CurFrameRes &oneFrame :
         m_multiCamFrames.at(curCamId).eachFrameInfo) {
      for (const std::pair<const int, std::vector<Eigen::Vector2d>> &oneData :
           oneFrame.mImagePointSets) {
        for (const auto &img_points : oneData.second) {
          int grid_col = floor(img_points.x() / static_cast<float>(gridPixels));
          int grid_row = floor(img_points.y() / static_cast<float>(gridPixels));
          assert(grid_col < gridWidth && grid_col >= 0);
          assert(grid_row < gridHeight && grid_row >= 0);
          // 出现一致，灰度值++
          if (image_gray.at<uint8_t>(grid_row, grid_col) <= 255)
            image_gray.at<uint8_t>(grid_row, grid_col)++;
          observedPointsNum++;
        }
      }
    }

    for (int row = 0; row < gridHeight; row++) {
      for (int col = 0; col < gridWidth; col++) {
        int observeTime = image_gray.at<uint8_t>(row, col);
        observeTime = observeTime * 255 / color_num;
        if (observeTime > 255)
          observeTime = (color_num - 1) * 255 / color_num;
        image_gray.at<uint8_t>(row, col) = observeTime;
      }
    }
    cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat image_dilate;
    dilate(image_gray, image_dilate, element);

    for (int row = 0; row < gridHeight; row++) {
      for (int col = 0; col < gridWidth; col++) {
        int gray_num = image_dilate.at<uint8_t>(row, col);
        if (gray_num > 0)
          coveragePointsNum++;
      }
    }

    cv::rectangle(image_dilate, cv::Point(gridWidth - 1, 0),
                  cv::Point(imageW - 1, gridHeight - 1),
                  cv::Scalar(255, 255, 255), -1, 4);
    cv::Scalar color[color_num];
    for (int i = 0; i < color_num; i++) {
      color[i] = cv::Scalar(i * 255 / color_num);
    }
    for (int i = 0; i < color_num; i++) {
      int h = 20, w = 80;
      int y = 50 * (i + 1);
      int x = gridWidth - 1 + 20;
      cv::rectangle(image_dilate, cv::Point(x, y), cv::Point(x + w, y + h),
                    color[i], -1);
      std::ostringstream txt_stream;
      if (i < color_num - 1)
        txt_stream << i << " points";
      else
        txt_stream << ">=" << i << " points";
      cv::putText(image_dilate, txt_stream.str(), cv::Point(x + w + 10, y + h),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0));
    }
    std::ostringstream txt_stream1, txt_stream2;
    txt_stream1 << std::to_string(curCamId) << " Coverage "
                << std::setprecision(4)
                << static_cast<float>(coveragePointsNum) /
                       static_cast<float>(gridWidth * gridHeight) * 100.0
                << "%";
    cv::putText(image_dilate, txt_stream1.str(),
                cv::Point(gridWidth - 1 + 20, gridHeight - 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0), 1);
    txt_stream2 << "Points Number: " << observedPointsNum;
    cv::putText(image_dilate, txt_stream2.str(),
                cv::Point(gridWidth - 1 + 20, gridHeight - 100),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0), 1);

    cv::imwrite(img_path, image_dilate);
  }
}

} // namespace dso