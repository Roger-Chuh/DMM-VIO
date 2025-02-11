#include "multi_camera_epipolar_search.h"
#include "estimator_config.h"

#include <opencv2/opencv.hpp>

//#define _SHOW_EPIPOLAR_SEARCH_DETAIL_
size_t searched_pid = dso::kInvalid;
// size_t searched_fid = 169;

namespace dso {

MultiCameraEpipolarSearch::MultiCameraEpipolarSearch(
    MultiCamera *cameras, const number_t &OOB_check_cos_theta_threshold,
    const number_t &cos_grad_epipolar_dir,
    const EstimatorConfig *estimator_config) {
  p_level_cid_to_camera_ = cameras;

  direct_visual_factor_.check_depth_ = false;
  //  direct_visual_factor_.check_dir_ = true;

  OOB_check_cos_theta_threshold_ = OOB_check_cos_theta_threshold;
  //  cos_grad_epipolar_dir_theta_threshold_ = cos_grad_epipolar_dir;

  search_target_level_ = estimator_config->search_level;
  number_t search_level_focal_length =
      p_level_cid_to_camera_->cid_to_cam[0]->GetParamByIndex(0) *
      std::pow(2.0f, -search_target_level_);

  rad_step_ = estimator_config->pixel_step *
              std::asin(1.0 / search_level_focal_length / 2.0) * 2.0;
}

MultiCameraEpipolarSearch::State MultiCameraEpipolarSearch::FindEpipolarMatch(
    const Point &point, const int &host_cid,
    std::array<std::shared_ptr<AlgsImage>, kCameraNumUsed> cid_to_img,
    const size_t &pid, const number_t &init_rho, const number_t &rho_sigma2,
    const size_t &target_fid,
    std::array<MatchRes, kCameraNumUsed> &cid_to_output, number_t &res_idp,
    const number_t &search_length_threshold, const bool &is_same_fid) {
  // Point point;
  const Patch &patch = point.pyramid_patch.patchs[0];
  //  VisualMeasurement host_vm;
  //  p_opt_database_->GetOrSetVM(point.host_vid, true, host_vm);
  //
  const size_t &host_fid = 1; // host_vm.fid;
  // const size_t &host_cid = 1; // host_vm.cid;
  //  NavState host_nav_state, target_nav_state;
  //  p_opt_database_->GetOrSetNavState(host_vm.fid, true, host_nav_state);
  //  p_opt_database_->GetOrSetNavState(target_fid, true, target_nav_state);
  // Mat4 Tcw0 = Mat4::Identity(); // = host_nav_state.v_Tcw[host_vm.cid];
  // Mat4 Twc0 = InversePose(Tcw0);

  // for factor evaluate
  Patch::ArrayV r_vec;
  number_t ws2, r2;

  number_t idp_min, idp_max;
  number_t idp_sigma = std::sqrt(rho_sigma2);
  idp_min = std::max(init_rho - idp_sigma, (number_t)0.01);
  idp_max = init_rho + idp_sigma;

  if (idp_max < idp_min || idp_max < 0) {
    //#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
    //    if (pid == searched_pid) {
    //    }
    //#endif
    return kFail;
  }

  for (size_t target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
    CamData &cur_search_data = cid_to_cam_data_[target_cid];
    cur_search_data.T10 =
        InversePose(p_level_cid_to_camera_->cid_to_T01.at(target_cid)) *
        p_level_cid_to_camera_->cid_to_T01.at(
            host_cid); // = target_nav_state.v_Tcw[target_cid] * Twc0;
    cur_search_data.T01 = InversePose(cur_search_data.T10);
    cur_search_data.target_level =
        search_target_level_; // todo: change target level

    CameraBase *camera =
        p_level_cid_to_camera_->cid_to_cam_pinhole.at(target_cid);
    //    std::cout << "width: " << camera->width()
    //              << ", height: " << camera->height()
    //              << ", patch dir0: " << patch.dir0.transpose()
    //              << ", point.n: " << point.n.transpose() << std::endl;
    // Check same fid and cid
    if (is_same_fid && target_cid == host_cid) {
      cur_search_data.state = kReject;
      continue;
    }

    // Check OOB
    {
      // Check mid dir
      cur_search_data.dir_mid =
          (cur_search_data.T10.block<3, 3>(0, 0) * point.n +
           init_rho * cur_search_data.T10.block<3, 1>(0, 3))
              .normalized();
      if (cur_search_data.dir_mid.z() < OOB_check_cos_theta_threshold_) {
        cur_search_data.state = kReject;
        continue;
      }
      camera->Project(cur_search_data.dir_mid, cur_search_data.uv_mid);
      if (!InFrame(cur_search_data.uv_mid, camera->width(), camera->height(),
                   1)) {
        cur_search_data.state = kReject;
        continue;
      }

      // Check left dir
      cur_search_data.dir_left =
          (cur_search_data.T10.block<3, 3>(0, 0) * point.n +
           idp_min * cur_search_data.T10.block<3, 1>(0, 3))
              .normalized();
      if (cur_search_data.dir_left.z() < OOB_check_cos_theta_threshold_) {
        cur_search_data.state = kReject;
        continue;
      }
      camera->Project(cur_search_data.dir_left, cur_search_data.uv_left);
      if (!InFrame(cur_search_data.uv_left, camera->width(), camera->height(),
                   1)) {
        cur_search_data.state = kReject;
        continue;
      }

      // Check right dir
      cur_search_data.dir_right =
          (cur_search_data.T10.block<3, 3>(0, 0) * point.n +
           idp_max * cur_search_data.T10.block<3, 1>(0, 3))
              .normalized();
      if (cur_search_data.dir_right.z() < OOB_check_cos_theta_threshold_) {
        cur_search_data.state = kReject;
        continue;
      }
      camera->Project(cur_search_data.dir_right, cur_search_data.uv_right);
      if (!InFrame(cur_search_data.uv_right, camera->width(), camera->height(),
                   1)) {
        cur_search_data.state = kReject;
        continue;
      }

      cur_search_data.epipolar_px_length =
          (cur_search_data.uv_left - cur_search_data.uv_right).norm();

      const number_t cur_cam_length_threshold =
          (host_cid == target_cid ? 5 * search_length_threshold
                                  : search_length_threshold);
      if (search_length_threshold > 0 &&
          cur_search_data.epipolar_px_length > cur_cam_length_threshold) {
        //        std::cout << "skip max px length " <<
        //        cur_search_data.epipolar_px_length << std::endl;
        cur_search_data.state = kReject;
        continue;
      }
    }

    cur_search_data.state = kVisible;
  }

  int visible_cam_num = 0;
  std::vector<size_t> searched_cid_vec;

  number_t cid_to_epipolar_length[kCameraNumUsed]; // = {-1, -1, -1, -1};
  for (int id = 0; id < kCameraNumUsed; ++id) {
    cid_to_epipolar_length[id] = -1;
  }

  for (size_t target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
    CamData &cur_search_data = cid_to_cam_data_[target_cid];
    if (cur_search_data.state == kVisible) {
      visible_cam_num++;
      searched_cid_vec.emplace_back(target_cid);
      cid_to_epipolar_length[target_cid] = cur_search_data.epipolar_px_length;
    }
    /*
    else if (cur_search_data.state == kParallel) {  // add the parallel cam to
    search searched_cid_vec.emplace_back(target_cid);
    }
     */
  }
  if (visible_cam_num == 0) {
#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
    if (pid == searched_pid) {
      printf("visible cam num is 0, Reject\n");
    }
#endif
    return kUnVisible;
  }

#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
  if (pid == searched_pid) {
    printf("cam state %d, %d, %d, %d\n", cid_to_cam_data_[0].state,
           cid_to_cam_data_[1].state, cid_to_cam_data_[2].state,
           cid_to_cam_data_[3].state);
  }
#endif

  const size_t &searched_cid = FindSearchedCid(cid_to_epipolar_length);

#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
  if (pid == searched_pid) {
    printf("epipolar length %f, %f, %f, %f, searched cid %zu, length %f\n",
           cid_to_epipolar_length[0], cid_to_epipolar_length[1],
           cid_to_epipolar_length[2], cid_to_epipolar_length[3], searched_cid,
           cid_to_epipolar_length[searched_cid]);
  }
#endif

  // do epipolar search
  MultiCamMatchRes best_search_res;
  bool has_success_match = false;
  bool has_valid_match = false;

  // calc left and right step
  CamData &searched_cam_data = cid_to_cam_data_[searched_cid];

  number_t cos_theta_left =
      searched_cam_data.dir_left.dot(searched_cam_data.dir_mid);
  cos_theta_left = cos_theta_left > 1.0f ? 1.0f : cos_theta_left;
  cos_theta_left = cos_theta_left < -1.0f ? -1.0f : cos_theta_left;
  number_t theta_left = std::acos(cos_theta_left);
  //  int n_left_steps =
  //      theta_left / rad_step_ > static_cast<number_t>(maxSteps_) ? maxSteps_
  //      : static_cast<int>(theta_left / rad_step_);

  number_t cos_theta_right =
      searched_cam_data.dir_mid.dot(searched_cam_data.dir_right);
  cos_theta_right = cos_theta_right > 1.0f ? 1.0f : cos_theta_right;
  cos_theta_right = cos_theta_right < -1.0f ? -1.0f : cos_theta_right;
  number_t theta_right = std::acos(cos_theta_right);
  //  int n_right_steps = theta_right / rad_step_ >
  //  static_cast<number_t>(maxSteps_)
  //                          ? maxSteps_
  //                          : static_cast<int>(theta_right / rad_step_);

  number_t total_theta = theta_left + theta_right;

  //  if (pid == searched_pid) {
  //    printf("pid %zu, n_left_step %d, n_right_step %d, left_step %f,
  //    right_step %f\n", pid, n_left_steps, n_right_steps,
  //           theta_left / rad_step_, theta_right / rad_step_);
  //  }

  direct_visual_factor_.calc_R_theta_ = false;
  if (total_theta > rad_step_ / 2) {
#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
    if (pid == searched_pid) {
      std::cout << "left steps " << (int)(theta_left / rad_step_)
                << " right steps " << (int)(theta_right / rad_step_)
                << " searched cam epi length "
                << searched_cam_data.epipolar_px_length << std::endl;
    }
#endif

    Vec3 normal_dir =
        searched_cam_data.dir_left.cross(searched_cam_data.dir_right);
    normal_dir.normalize();

    Mat3 R_step = ExpSO3(normal_dir * rad_step_);
    Mat3 R_step_T = R_step.transpose();
    //    Mat3 R_to_start = ExpSO3(-normal_dir * rad_step_ * n_left_steps);
    //    Vec3 search_dir = R_to_start * searched_cam_data.dir_mid;

#if CODE_ACC_SEARCH_DIR_VEC
    aligned_vector<Vec3> search_dir_vec;
    number_t left_step_number_t = theta_left / rad_step_;
    int left_step = static_cast<int>(left_step_number_t);
    int left_bool =
        left_step < maxSteps_ && left_step_number_t - (number_t)left_step > 0.1;

    number_t right_step_number_t = theta_right / rad_step_;
    int right_step = static_cast<int>(right_step_number_t);
    int right_bool = right_step < maxSteps_ &&
                     right_step_number_t - (number_t)right_step > 0.1;

    int search_dir_vec_num = std::min(maxSteps_, left_step) + left_bool + 1 +
                             std::min(maxSteps_, right_step) + right_bool;
    search_dir_vec.resize(search_dir_vec_num);
    int search_dir_vec_i = std::min(maxSteps_, left_step) + left_bool - 1;
    Vec3 *psearch_dir_vec_push =
        &search_dir_vec[std::min(maxSteps_, left_step) + left_bool - 1];
    {
      Vec3 search_dir = searched_cam_data.dir_mid;

      int i = 0;
      for (; i < std::min(maxSteps_, left_step); ++i) {
        search_dir = R_step_T * search_dir;
        search_dir_vec[search_dir_vec_i--] = search_dir;
      }
      if (left_bool) {
        search_dir_vec[search_dir_vec_i--] = searched_cam_data.dir_left;
      }
    }

    search_dir_vec_i = std::min(maxSteps_, left_step) + left_bool;

    search_dir_vec[search_dir_vec_i++] = searched_cam_data.dir_mid;

    {
      Vec3 search_dir = searched_cam_data.dir_mid;

      int i = 0;
      for (; i < std::min(maxSteps_, right_step); ++i) {
        search_dir = R_step * search_dir;
        search_dir_vec[search_dir_vec_i++] = search_dir;
      }
      if (right_bool) {
        search_dir_vec[search_dir_vec_i++] = searched_cam_data.dir_right;
      }
    }
#else  // CODE_ACC_SEARCH_DIR_VEC
    aligned_vector<Vec3> search_dir_vec;
    {
      number_t left_step_number_t = theta_left / rad_step_;
      int left_step = static_cast<int>(left_step_number_t);
      Vec3 search_dir = searched_cam_data.dir_mid;

      int i = 0;
      for (; i < std::min(maxSteps_, left_step); ++i) {
        search_dir = R_step_T * search_dir;
        search_dir_vec.emplace_back(search_dir);
      }
      if (left_step < maxSteps_ &&
          left_step_number_t - (number_t)left_step > 0.1) {
        search_dir_vec.emplace_back(searched_cam_data.dir_left);
      }

      std::reverse(search_dir_vec.begin(), search_dir_vec.end());
    }

    search_dir_vec.emplace_back(searched_cam_data.dir_mid);

    {
      number_t right_step_number_t = theta_right / rad_step_;
      int right_step = static_cast<int>(right_step_number_t);
      Vec3 search_dir = searched_cam_data.dir_mid;

      int i = 0;
      for (; i < std::min(maxSteps_, right_step); ++i) {
        search_dir = R_step * search_dir;
        search_dir_vec.emplace_back(search_dir);
      }
      if (right_step < maxSteps_ &&
          right_step_number_t - (number_t)right_step > 0.1) {
        search_dir_vec.emplace_back(searched_cam_data.dir_right);
      }
    }
#endif // CODE_ACC_SEARCH_DIR_VEC

#if CODE_ACC_Triangulate

    int Triangulate_size = search_dir_vec.size();
    number_t idp_array[Triangulate_size + 4];
    void Triangulate_ACC_array(float *p_idp, const float *pM4_T01,
                               const float *pV3_v0, const float *pV3_v1,
                               int size);
    Triangulate_ACC_array(
        &idp_array[0], &cid_to_cam_data_[searched_cid].T01(0, 0),
        &point.n(0, 0), &search_dir_vec[0](0, 0), Triangulate_size);

#endif
    //    std::cout << "search dir vec size " << search_dir_vec.size() <<
    //    std::endl;

    // todo: cal all dir by step, left and right dir

    size_t max_match_cam_num = 0;
    number_t max_avg_zncc = -1;
    int max_index = -1;

    size_t second_max_match_cam_num = 0;
    number_t second_max_avg_zncc = 0;
    int second_max_index = -1;

    /*
    if (n_left_steps + n_right_steps + 1 < 0) {
      printf("length err %d\n", n_left_steps + n_right_steps + 1);
      printf("theta_left %f, theta_right %f\n", theta_left, theta_right);
      std::cout << "cos theta left " << cos_theta_left << " dir left " <<
    searched_cam_data.dir_left.transpose()
                << " norm " << searched_cam_data.dir_left.norm() << " dir mid "
    << searched_cam_data.dir_mid.transpose()
                << " norm " << searched_cam_data.dir_mid.norm() << std::endl;

      std::cout << "cos theta right " << cos_theta_right << " dir right " <<
    searched_cam_data.dir_right.transpose()
                << " norm " << searched_cam_data.dir_right.norm() << " dir mid "
                << searched_cam_data.dir_mid.transpose() << std::endl;
      std::abort();
    }
     */

    size_t total_step = search_dir_vec.size();
    std::vector<MultiCamMatchRes> search_res_vec(total_step);
    for (int i = 0; i < total_step; ++i) {
      MultiCamMatchRes &multi_cam_match_res = search_res_vec[i];

#if CODE_ACC_Triangulate
      multi_cam_match_res.idp = idp_array[i];
      multi_cam_match_res.tri_success = (multi_cam_match_res.idp != -1.0f);
#else
      multi_cam_match_res.tri_success = Triangulate(
          multi_cam_match_res.idp, cid_to_cam_data_[searched_cid].T01, point.n,
          search_dir_vec[i]);
#endif

      if (!multi_cam_match_res.tri_success || multi_cam_match_res.idp > 20.0) {
        multi_cam_match_res.match_success_cam_num = 0;
        multi_cam_match_res.avg_zncc = -1;
#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
        if (pid == searched_pid) {
          printf("index %d, idp %f, target cid %zu, tri fail\n", i,
                 multi_cam_match_res.idp, searched_cid);
        }
#endif
        continue;
      }

      for (const size_t &target_cid : searched_cid_vec) {
        CamData &cur_search_data = cid_to_cam_data_[target_cid];
        MatchRes &match_res = multi_cam_match_res.cid_to_match_res[target_cid];
        //        std::shared_ptr<AlgsImage> target_image =
        //            target_nav_state.cid_level_to_img[target_cid][cur_search_data.target_level];
        //        direct_visual_factor_.cur_target_level_ = 0;
        //        direct_visual_factor_.target_image_level0_ =
        //        target_nav_state.cid_level_to_img[target_cid][0];
        CameraBase *camera =
            p_level_cid_to_camera_->cid_to_cam_pinhole.at(target_cid);
        DirectFactorRes direct_factor_res = direct_visual_factor_.Evaluate(
            cur_search_data.T10, multi_cam_match_res.idp,
            cid_to_img[target_cid], patch, camera, 1, r_vec, ws2, r2,
            &match_res.target_uv, &match_res.target_dir,
            &match_res.disparity_cos_theta, nullptr, &match_res.zncc,
            search_zncc_threshold_);

#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
        if (pid == searched_pid /*&& target_cid == 3 &&
            (direct_factor_res == kGoodInlier || direct_factor_res == kPassInlier)*/) {
          printf(
              "index %d, idp %f, target cid %zu, res %d, zncc %f, "
              "dir_reject %d, epipolar %f, disparity %f\n",
              i, multi_cam_match_res.idp, target_cid, (int)direct_factor_res,
              match_res.zncc, direct_visual_factor_.dir_reject_,
              std::acos(direct_visual_factor_.epipolar_grad_cos_theta_) *
                  180.0 / M_PI,
              //              std::acos(direct_visual_factor_.test_grad_by_dir_cos_theta_)
              //              * 180.0 / M_PI,
              match_res.disparity_cos_theta);
        }
#endif

        if (direct_factor_res == DirectFactorRes::kInlier) {
          has_success_match = true;
          match_res.match_success = true;
          match_res.epipolar_grad_cos_theta =
              direct_visual_factor_.epipolar_grad_cos_theta_;
          match_res.dir_reject = direct_visual_factor_.dir_reject_;
          if (!match_res.dir_reject) {
            has_valid_match = true;
            multi_cam_match_res.match_success_cam_num++;
            multi_cam_match_res.avg_zncc += match_res.zncc;
          }
        } else {
          match_res.match_success = false;
        }
      }

      multi_cam_match_res.avg_zncc /=
          (number_t)multi_cam_match_res.match_success_cam_num;

      if (multi_cam_match_res.match_success_cam_num > max_match_cam_num ||
          (multi_cam_match_res.match_success_cam_num == max_match_cam_num &&
           multi_cam_match_res.avg_zncc > max_avg_zncc)) {
        second_max_match_cam_num = max_match_cam_num;
        second_max_avg_zncc = max_avg_zncc;
        second_max_index = max_index;

        max_match_cam_num = multi_cam_match_res.match_success_cam_num;
        max_avg_zncc = multi_cam_match_res.avg_zncc;
        max_index = i;
      } else if (multi_cam_match_res.match_success_cam_num >
                     second_max_match_cam_num ||
                 (multi_cam_match_res.match_success_cam_num ==
                      second_max_match_cam_num &&
                  multi_cam_match_res.avg_zncc > second_max_avg_zncc)) {
        second_max_match_cam_num = multi_cam_match_res.match_success_cam_num;
        second_max_avg_zncc = multi_cam_match_res.avg_zncc;
        second_max_index = i;
      }
    }

    //    for (int i = 0; i < search_res_vec.size(); ++i) {
    //      std::cout << "index " << i << " success cam " <<
    //      search_res_vec[i].match_success_cam_num << " avg zncc "
    //                << search_res_vec[i].avg_zncc << " uv "
    //                <<
    //                search_res_vec[i].cid_to_match_res[searched_cid].target_uv.transpose()
    //                << std::endl;
    //    }

#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
    if (pid == searched_pid) {
      std::cout << "max match cam num " << max_match_cam_num << " max avg zncc "
                << max_avg_zncc << " max index " << max_index
                << " max match idp " << search_res_vec[max_index].idp
                << std::endl;
    }
#endif

#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_

    if (pid == searched_pid) {
      // show the search result
      {
        // draw host patch
        //        cv::Mat host_mat;
        //        cv::Mat host_gray_mat;
        //        std::shared_ptr<AlgsImage> p_img =
        //        host_nav_state.cid_level_to_img[host_cid][0]; host_gray_mat =
        //        cv::Mat(p_img->height, p_img->width, CV_8UC1, p_img->data,
        //        p_img->stride); cv::cvtColor(host_gray_mat, host_mat,
        //        cv::COLOR_GRAY2BGR);

        // draw target search result
        std::vector<cv::Mat> colored_mat_vec(4);
        cv::Mat gray_mat;
        for (size_t cam_id = 0; cam_id < kCameraNumUsed; ++cam_id) {
          std::shared_ptr<AlgsImage> p_img =
              target_nav_state.cid_level_to_img[cam_id][search_target_level_];
          gray_mat = cv::Mat(p_img->height, p_img->width, CV_8UC1, p_img->data,
                             p_img->stride);
          cv::cvtColor(gray_mat, colored_mat_vec[cam_id], cv::COLOR_GRAY2BGR);
        }

        size_t test_index = 0;
        for (const MultiCamMatchRes &multi_cam_match_res : search_res_vec) {
          //          printf("index %zu, tri success %d, success num %zu, avg
          //          zncc %f\n", test_index,
          //                 multi_cam_match_res.tri_success,
          //                 multi_cam_match_res.match_success_cam_num,
          //                 multi_cam_match_res.avg_zncc);
          test_index++;
          if (!multi_cam_match_res.tri_success) {
            continue;
          }

          for (size_t cam_id = 0; cam_id < kCameraNumUsed; ++cam_id) {
            const MatchRes match_res =
                multi_cam_match_res.cid_to_match_res[cam_id];
            if (match_res.match_success) {
              colored_mat_vec[cam_id].at<cv::Vec3b>(match_res.target_uv[1],
                                                    match_res.target_uv[0]) =
                  cv::Vec3b(0, 255, 0);
            } else {
              colored_mat_vec[cam_id].at<cv::Vec3b>(match_res.target_uv[1],
                                                    match_res.target_uv[0]) =
                  cv::Vec3b(0, 0, 255);
            }

            //            if (target_fid == searched_fid && cam_id != host_cid)
            //            {
            //              const number_t& searched_idp =
            //              multi_cam_match_res.idp; Vec3 searched_dir =
            //              test_host_dir / searched_idp;
            //
            //              Mat4 T10 = target_nav_state.v_Tcw[cam_id] *
            //              InversePose(target_nav_state.v_Tcw[host_cid]); Vec3
            //              target_searched_dir = T10.block<3, 3>(0, 0) *
            //              searched_dir + T10.block<3, 1>(0, 3);
            //              CameraBase::Ptr camera =
            //              level_cid_to_camera_[0]->GetCam(cam_id); Vec2
            //              target_searched_uv;
            //              camera->Project(target_searched_dir,
            //              target_searched_uv);
            //              colored_mat_vec[cam_id].at<cv::Vec3b>(target_searched_uv[1],
            //              target_searched_uv[0]) =
            //                  cv::Vec3b(255, 0, 0);
            //            }
          }
        }

        // draw max match result
        std::cout << "max match before draw, cam num "
                  << search_res_vec[max_index].match_success_cam_num
                  << " avg zncc " << search_res_vec[max_index].avg_zncc
                  << std::endl;
        for (size_t cam_id = 0; cam_id < kCameraNumUsed; ++cam_id) {
          const MatchRes match_res =
              search_res_vec[max_index].cid_to_match_res[cam_id];
          if (match_res.match_success) {
            cv::circle(
                colored_mat_vec[cam_id],
                cv::Point(match_res.target_uv[0], match_res.target_uv[1]), 2,
                cv::Scalar(0, 255, 0), -1);
          } else {
            cv::circle(
                colored_mat_vec[cam_id],
                cv::Point(match_res.target_uv[0], match_res.target_uv[1]), 2,
                cv::Scalar(0, 0, 255), -1);
          }
        }

        cv::Mat temp1, temp2, res;
        cv::vconcat(colored_mat_vec[1], colored_mat_vec[0], temp1);
        cv::vconcat(colored_mat_vec[2], colored_mat_vec[3], temp2);
        cv::hconcat(temp1, temp2, res);
        cv::imshow("MultiCamSearch Result", res);

        //        cv::imshow("Host Patch", host_mat);
      }
    }

#endif

    if (!has_success_match) {
#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
      if (pid == searched_pid) {
        printf("multi camera epipolar search fail\n");
        cv::waitKey(0);
      }
#endif
      return kFail;
    }

    if (!has_valid_match) {
#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
      if (pid == searched_pid) {
        printf("all the success match cam is parallel\n");
        cv::waitKey(0);
      }
#endif
      return kReject;
    }

    best_search_res = search_res_vec[max_index];

    if (max_match_cam_num == second_max_match_cam_num &&
        max_avg_zncc - second_max_avg_zncc < 0.01 &&
        std::abs(max_index - second_max_index) > 2) {
      return kReject;
    }

    /*
    else if (success_visible_cam == 1) {
      // check ambiguous
      std::vector<number_t> zncc_vec(search_res_vec.size(), -1);
      for (size_t index = 0; index < search_res_vec.size(); ++index) {
        if (!search_res_vec[index].tri_success) {
          continue;
        }
        if (search_res_vec[index].match_success_cam_num != max_match_cam_num) {
          continue;
        }
        zncc_vec[index] = search_res_vec[index].avg_zncc;
      }

      std::vector<size_t> maxima_index_vec = FindLocalMaxima(zncc_vec);

#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
      if (pid == searched_pid) {
        std::cout << "zncc vec ";
        for (const number_t& zncc : zncc_vec) {
          std::cout << zncc << " ";
        }
        std::cout << std::endl;

        std::cout << "maxima index ";
        for (const size_t& maxima_index : maxima_index_vec) {
          std::cout << maxima_index << " ";
        }
        std::cout << std::endl;
      }
#endif

      // when there are two or more local maxima, check the best score and the
second score if (maxima_index_vec.size() >= 2) { number_t best_zncc,
second_zncc; GetTheBestAndSecondScore(zncc_vec, maxima_index_vec, best_zncc,
second_zncc); printf("pid %zu, match success cam num %zu, best zncc %f, second
zncc %f\n", pid, search_success_cid_vec.size(), best_zncc, second_zncc);

        if (best_zncc - second_zncc < 0.0) {
#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
          if (pid == searched_pid) {
            printf("best zncc %f, second zncc %f, reject\n", best_zncc,
second_zncc); cv::waitKey(0);
          }
#endif
          return kReject;
        }
      }
    }
    */
  } else {
    best_search_res.tri_success = true;
    best_search_res.idp = init_rho;
    for (const size_t &target_cid : searched_cid_vec) {
      CamData &cur_search_data = cid_to_cam_data_[target_cid];
      MatchRes &match_res = best_search_res.cid_to_match_res[target_cid];
      //      std::shared_ptr<AlgsImage> target_image =
      //          target_nav_state.cid_level_to_img[target_cid][cur_search_data.target_level];
      //      direct_visual_factor_.cur_target_level_ = 0;
      //      direct_visual_factor_.target_image_level0_ =
      //      target_nav_state.cid_level_to_img[target_cid][0];
      CameraBase *camera =
          p_level_cid_to_camera_->cid_to_cam_pinhole.at(target_cid);
      DirectFactorRes direct_factor_res = direct_visual_factor_.Evaluate(
          cur_search_data.T10, best_search_res.idp, cid_to_img[target_cid],
          patch, camera, 1, r_vec, ws2, r2, &match_res.target_uv,
          &match_res.target_dir, &match_res.disparity_cos_theta, nullptr,
          &match_res.zncc, search_zncc_threshold_);

      if (direct_factor_res == kInlier) {
        has_success_match = true;
        match_res.match_success = true;
        match_res.epipolar_grad_cos_theta =
            direct_visual_factor_.epipolar_grad_cos_theta_;
        match_res.dir_reject = direct_visual_factor_.dir_reject_;

        if (!match_res.dir_reject) {
          has_valid_match = true;
          best_search_res.match_success_cam_num++;
          best_search_res.avg_zncc += match_res.zncc;
        }
      } else {
        match_res.match_success = false;
      }
    }

    if (!has_success_match) {
#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
      if (pid == searched_pid) {
        printf("epipolar length < 1.0 and search fail\n");
      }
#endif
      return kFail;
    }

    if (!has_valid_match) {
#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
      if (pid == searched_pid) {
        printf("all the success match cam is parallel\n");
        cv::waitKey(0);
      }
#endif
      return kReject;
    }

    best_search_res.avg_zncc /= (number_t)best_search_res.match_success_cam_num;
  }

  //  std::vector<size_t> search_success_cid_vec;
  std::vector<size_t> opt_cid_vec;
  for (size_t cid = 0; cid < kCameraNumUsed; ++cid) {
    if (best_search_res.cid_to_match_res[cid].match_success && !best_search_res.cid_to_match_res[cid].dir_reject/* &&
        best_search_res.cid_to_match_res[cid].disparity_cos_theta < 0.99999*/) {
      opt_cid_vec.emplace_back(cid);
      //      search_success_cid_vec.emplace_back(cid);
      //      if (!best_search_res.cid_to_match_res[cid].dir_reject &&
      //          best_search_res.cid_to_match_res[cid].disparity_cos_theta <
      //          0.999848) {
      //        success_visible_cam++;
      //      }
    }
  }

  if (opt_cid_vec.empty()) {
#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
    if (pid == searched_pid) {
      printf("all the success match cam, disparity is too small\n");
      cv::waitKey(0);
    }
#endif
    return kReject;
  }

  //  std::vector<size_t> opt_cid_vec;
  //  for (size_t cid = 0; cid < kCameraNumUsed; ++cid) {
  //    if (best_search_res.cid_to_match_res[cid].match_success) {
  //      opt_cid_vec.emplace_back(cid);
  //    }
  //  }

  direct_visual_factor_.calc_R_theta_ = true;

  number_t lambda = 0.1;
  const number_t lambdaSuccessFac = 0.5;
  const number_t lambdaFailFac = 5.0;
  const number_t convergenceEps = 1e-3;

  size_t success_step = 0;

  number_t cur_idp = best_search_res.idp;
  number_t H22 = 0, b2 = 0;
  number_t energy = 0;
  size_t success_vm_num = 0;
  Patch::VectorV J_idp;
  Vec3 dp_didp;

  bool opt_converged = false;

  number_t before_opt_zncc_sum = 0;
  number_t after_opt_zncc_sum = 0;

  for (const size_t &target_cid : opt_cid_vec) {
    CamData &cur_search_data = cid_to_cam_data_[target_cid];
    //    std::shared_ptr<AlgsImage> target_image =
    //    target_nav_state.cid_level_to_img[target_cid][opt_target_level_];
    size_t target_level = 1;
    CameraBase *camera =
        p_level_cid_to_camera_->cid_to_cam_pinhole.at(target_cid);

    MatchRes &match_res = cid_to_output[target_cid];
    match_res.match_success = true;
    match_res.T01 = cid_to_cam_data_[target_cid].T01;

    //    direct_visual_factor_.cur_target_level_ = 0;
    //    direct_visual_factor_.target_image_level0_ =
    //    target_nav_state.cid_level_to_img[target_cid][0];

    DirectFactorRes direct_factor_res = direct_visual_factor_.Evaluate(
        cur_search_data.T10, cur_idp, cid_to_img[target_cid], patch, camera, 1,
        r_vec, ws2, r2, &(match_res.target_uv), &(match_res.target_dir),
        nullptr, &dp_didp, &(match_res.zncc), opt_zncc_threshold_);

    if (direct_factor_res != kInlier) {
      //      LOG_VIO_INFO("pid %zu, target cid %d, pre zncc %f, cur res %d, cur
      //      zncc %f\n", pid, target_cid,
      //                   best_search_res.cid_to_match_res[target_cid].zncc,
      //                   (int)direct_factor_res, match_res.zncc);
      continue;
    }

    match_res.R_theta = direct_visual_factor_.R_theta_;

    before_opt_zncc_sum += match_res.zncc;

    success_vm_num++;
    /*
    //#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
    //    if (pid == searched_pid) {
    //      printf("opt target cid %zu, res %d, zncc %f, epipolar_grad_cos_theta
    %f, dir reject %d\n", target_cid,
    //             (int)direct_factor_res, match_res.zncc,
    direct_visual_factor_.epipolar_grad_cos_theta_,
    //             (int)direct_visual_factor_.dir_reject_);
    //    }
    //#endif
    */

    match_res.epipolar_grad_cos_theta =
        direct_visual_factor_.epipolar_grad_cos_theta_;

    J_idp = patch.J_dir * dp_didp;

    // camera->Project(patch.dir0,)
    // std::cout << "patch.J_dir:\n " << patch.J_dir.transpose() << "\ndp_didp:
    // " << dp_didp.transpose()<< ", J_idp: " << J_idp.transpose() << ", r_vec:
    // "
    // << r_vec.matrix().transpose() << std::endl;
    H22 += ws2 * J_idp.transpose() * J_idp;
    b2 += ws2 * J_idp.transpose() * r_vec.matrix();

    energy += ws2 * r2;
  }

  if (success_vm_num != opt_cid_vec.size()) {
    //    LOG_VIO_ERROR("VIO Error, success vm num less than opt_cid_vec
    //    size\n");
    return kFail;
  }

  for (int iter = 0; iter < max_iter_; ++iter) {
    H22 *= 1.0 + lambda;
    const number_t step = -b2 / H22;
    // printf("step: %f, b2: %f, H22: %f, success_vm_num: %d\n", step, b2, H22,
    // success_vm_num);
    const number_t new_idp = cur_idp + step;

    if (new_idp < 0) {
      break;
    }

    number_t new_H22 = 0;
    number_t new_b2 = 0;
    number_t new_energy = 0;
    size_t new_success_vm_num = 0;

    std::array<MatchRes, kCameraNumUsed> new_cid_to_output;

    for (const size_t &target_cid : opt_cid_vec) {
      CamData &cur_search_data = cid_to_cam_data_[target_cid];
      //      std::shared_ptr<AlgsImage> target_image =
      //      target_nav_state.cid_level_to_img[target_cid][opt_target_level_];

      CameraBase *camera =
          p_level_cid_to_camera_->cid_to_cam_pinhole.at(target_cid);

      MatchRes &match_res = new_cid_to_output[target_cid];
      match_res.match_success = true;
      match_res.T01 = cid_to_cam_data_[target_cid].T01;

      //      direct_visual_factor_.cur_target_level_ = 0;
      //      direct_visual_factor_.target_image_level0_ =
      //      target_nav_state.cid_level_to_img[target_cid][0];

      DirectFactorRes direct_factor_res = direct_visual_factor_.Evaluate(
          cur_search_data.T10, new_idp, cid_to_img[target_cid], patch, camera,
          1, r_vec, ws2, r2, &(match_res.target_uv), &(match_res.target_dir),
          nullptr, &dp_didp, &(match_res.zncc), opt_zncc_threshold_);

      match_res.R_theta = direct_visual_factor_.R_theta_;

      if (direct_factor_res != kInlier) {
        break;
      }

      new_success_vm_num++;
      /*
      //#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
      //    if (pid == searched_pid) {
      //      printf("opt target cid %zu, res %d, zncc %f,
      epipolar_grad_cos_theta %f, dir reject %d\n", target_cid,
      //             (int)direct_factor_res, match_res.zncc,
      direct_visual_factor_.epipolar_grad_cos_theta_,
      //             (int)direct_visual_factor_.dir_reject_);
      //    }
      //#endif
      */

      match_res.epipolar_grad_cos_theta =
          direct_visual_factor_.epipolar_grad_cos_theta_;

      J_idp = patch.J_dir * dp_didp;

      new_H22 += ws2 * J_idp.transpose() * J_idp;
      new_b2 += ws2 * J_idp.transpose() * r_vec.matrix();

      new_energy += ws2 * r2;
    }

    if (new_energy < energy && new_success_vm_num == opt_cid_vec.size()) {
      // save new values
      cur_idp = new_idp;
      H22 = new_H22;
      b2 = new_b2;
      energy = new_energy;
      cid_to_output = new_cid_to_output;

      success_step++;

      lambda *= lambdaSuccessFac;
    } else {
      lambda *= lambdaFailFac;
    }

    if (std::fabs(step) < convergenceEps) {
      opt_converged = true;
      break;
    }
  }

  for (size_t cid = 0; cid < kCameraNumUsed; ++cid) {
    if (cid_to_output[cid].match_success) {
      after_opt_zncc_sum += cid_to_output[cid].zncc;
    }
  }

#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
  if (pid == searched_pid) {
    for (size_t cid = 0; cid < kCameraNumUsed; ++cid) {
      printf("target cid %zu, match success %d, dir %f, %f, %f, zncc %f, "
             "epipolar_grad_cos_theta %f\n",
             cid, (int)cid_to_output[cid].match_success,
             cid_to_output[cid].target_dir[0], cid_to_output[cid].target_dir[1],
             cid_to_output[cid].target_dir[2], cid_to_output[cid].zncc,
             cid_to_output[cid].epipolar_grad_cos_theta);
    }

    printf("pid %zu, opt cam num %zu, zncc before %f, after %f, idp before %f, "
           "after %f, success step %zu, idp is "
           "converged %d, opt success %d\n",
           pid, opt_cid_vec.size(), before_opt_zncc_sum / opt_cid_vec.size(),
           after_opt_zncc_sum / opt_cid_vec.size(), best_search_res.idp,
           cur_idp, success_step, (int)opt_converged,
           (success_step >= 4 || opt_converged) ? 1 : 0);
  }
#endif
  //  if (success_step >= 4 || opt_converged) {
  //    cv::waitKey(1);
  //  } else {
  //    cv::waitKey(0);
  //  }
#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
  if (pid == searched_pid) {
    cv::waitKey(0);
  }
#endif

  if (success_step >= 4 || opt_converged) {
    res_idp = cur_idp;
    return kSuccess;
  } else {
    return kFail;
  }
}

bool MultiCameraEpipolarSearch::InFrame(const Vec2 &uv, const size_t &img_width,
                                        const size_t &img_height,
                                        const int &border) {
  if (uv[0] >= border && uv[0] < img_width - border && uv[1] >= border &&
      uv[1] < img_height - border) {
    return true;
  }
  return false;
}

bool MultiCameraEpipolarSearch::Triangulate(number_t &idp, const Mat4 &T01,
                                            const Vec3 &v0, const Vec3 &v1) {
  Mat63 A = Mat63::Zero();
  Vec6 b = Vec6::Zero();
  A.block<3, 3>(0, 0) = Skew(v0);
  A.block<3, 3>(3, 0) = Skew(v1) * T01.block<3, 3>(0, 0).transpose();
  b.segment<3>(3) =
      Skew(v1) * T01.block<3, 3>(0, 0).transpose() * T01.block<3, 1>(0, 3);
  Vec3 s = A.transpose() * b;
  Mat3 AA = A.transpose() * A;
  Vec3 xyz = AA.ldlt().solve(s);

  if (xyz.z() < 0) {
    return false;
  }
  idp = 1.0 / xyz.norm();

  Vec3 v1_0 = T01.block<3, 3>(0, 0) * v1;
  number_t cos_theta = v1_0.dot(v0);
  //  if (cos_theta > 0.99999) {
  //    return false;
  //  }

  return true;
}

size_t MultiCameraEpipolarSearch::FindSearchedCid(
    const number_t *cid_to_epipolar_length) {
  //  number_t minAbove = std::numeric_limits<number_t>::max();
  //  size_t minAboveCid = 0;
  //  number_t maxBelow = -1.0;
  //  size_t maxBelowCid = 0;
  //
  //  for (size_t cid = 0; cid < kCameraNumUsed; ++cid) {
  //    if (cid_to_epipolar_length[cid] < 0) {
  //      continue;
  //    }
  //
  //    const number_t& cur_length = cid_to_epipolar_length[cid];
  //    if (cur_length > epipolar_length_threshold_ && cur_length < minAbove) {
  //      minAbove = cur_length;
  //      minAboveCid = cid;
  //    } else if (cur_length <= epipolar_length_threshold_ && cur_length >
  //    maxBelow) {
  //      maxBelow = cur_length;
  //      maxBelowCid = cid;
  //    }
  //  }
  //
  //  if (minAbove < std::numeric_limits<number_t>::max()) {
  //    return minAboveCid;
  //  } else {
  //    return maxBelowCid;
  //  }

  number_t max_length = 0;
  size_t max_length_cid = 0;
  for (size_t cid = 0; cid < kCameraNumUsed; ++cid) {
    if (cid_to_epipolar_length[cid] < 0) {
      continue;
    }
    const number_t &cur_length = cid_to_epipolar_length[cid];
    if (cur_length > max_length) {
      max_length = cur_length;
      max_length_cid = cid;
    }
  }

  return max_length_cid;
}

std::vector<size_t> MultiCameraEpipolarSearch::FindLocalMaxima(
    const std::vector<number_t> &zncc_vec) {
  std::vector<size_t> maxima_index_vec;
  if (zncc_vec[0] > zncc_vec[1]) {
    maxima_index_vec.emplace_back(0);
  }

  for (int i = 1; i < zncc_vec.size() - 1; ++i) {
    if (zncc_vec[i] > zncc_vec[i - 1] && zncc_vec[i] > zncc_vec[i + 1]) {
      maxima_index_vec.emplace_back(i);
    }
  }

  size_t last_index = zncc_vec.size() - 1;
  if (zncc_vec[last_index] > zncc_vec[last_index - 1]) {
    maxima_index_vec.emplace_back(last_index);
  }

  return maxima_index_vec;
}

void MultiCameraEpipolarSearch::GetTheBestAndSecondScore(
    const std::vector<number_t> &zncc_vec,
    const std::vector<size_t> &maxima_index_vec, number_t &best_zncc,
    number_t &second_zncc) {
  best_zncc = -1;
  second_zncc = -1;
  //  size_t best_zncc_index = -1;
  //  size_t second_best_zncc_index = -1;

  for (const size_t &index : maxima_index_vec) {
    const number_t &cur_zncc = zncc_vec[index];
    if (cur_zncc > best_zncc) {
      second_zncc = best_zncc;
      //      second_best_zncc_index = best_zncc_index;

      best_zncc = cur_zncc;
      //      best_zncc_index = index;
    } else if (cur_zncc > second_zncc && cur_zncc < best_zncc) {
      second_zncc = cur_zncc;
      //      second_best_zncc_index = index;
    }
  }
}

#if CODE_ACC_Triangulate

#if CODE_ACC_Triangulate >= 3
void ldlt_3_simd4(const float *pm3A, const float *pv3B, float *pv3X) {
#define A_INFO()
#define X_INFO()
#define OUT_INFO()

  bool found_zero_pivot = false;
  bool ret = true;

  float m3A00 = pm3A[0];
  float m3A01 = pm3A[1];
  float m3A02 = pm3A[2];

  float m3A10 = pm3A[3];
  float m3A11 = pm3A[4];
  float m3A12 = pm3A[5];

  float m3A20 = pm3A[6];
  float m3A21 = pm3A[7];
  float m3A22 = pm3A[8];

  float v3B0 = pv3B[0];
  float v3B1 = pv3B[1];
  float v3B2 = pv3B[2];

  float v3X0;
  float v3X1;
  float v3X2;

  float min_f32 = (std::numeric_limits<float>::min)();
  float imm_0_f32 = 0;

  // unblocked
  {// unblocked::for(k=0:3)
   {
// A21 = m3A10,m3A20
// A10 =
// A20 =
#define MAX_SWAP 0
#if MAX_SWAP
       int max_index = {m3A00, m3A11, m3A22};
  if (max_index != 0) {
    swap
  }
#endif

  float realAkk = m3A00;
  bool pivot_is_valid = (abs(realAkk) > 0.0f);
#if 0
      if(k==0 && !pivot_is_valid){some code,return;}
#endif

  if (pivot_is_valid) {
    m3A10 /= realAkk;
    m3A20 /= realAkk;
  } else {
    ret = ret && ((m3A10 == 0) && (m3A20 == 0));
  }

  if (found_zero_pivot && pivot_is_valid)
    ret = false; // factorization failed
  else if (!pivot_is_valid)
    found_zero_pivot = true;

#if 0
  	sign code
#endif

  A_INFO();
}

// unblocked::for(k=1:3)
{
// A21 = m3A21
// A10 = m3A10
// A20 = m3A20
#define MAX_SWAP 0
#if MAX_SWAP
  int max_index = {m3A00, m3A11, m3A22};
  if (max_index != 0) {
    swap
  }
#endif

  float temp = m3A00 * m3A10;
  m3A11 -= m3A10 * temp;
  m3A21 -= m3A20 * temp;

  float realAkk = m3A11;
  bool pivot_is_valid = (abs(realAkk) > 0.0f);

  if (pivot_is_valid) {
    m3A21 /= realAkk;
  } else {
    ret = ret && (m3A21 == 0);
  }

  if (found_zero_pivot && pivot_is_valid)
    ret = false; // factorization failed
  else if (!pivot_is_valid)
    found_zero_pivot = true;

#if 0
  	sign code
#endif

  A_INFO();
}

// unblocked::for(k=2:3)
{
// A21 =
// A10 = m3A20 m3A21
// A20 =
#define MAX_SWAP 0
#if MAX_SWAP
  int max_index = {m3A00, m3A11, m3A22};
  if (max_index != 0) {
    swap
  }
#endif

  float temp0 = m3A00 * m3A20;
  float temp1 = m3A11 * m3A21;
  m3A22 -= m3A20 * temp0 + m3A21 * temp1;

  float realAkk = m3A22;
  bool pivot_is_valid = (abs(realAkk) > 0.0f);

  if (found_zero_pivot && pivot_is_valid)
    ret = false; // factorization failed
  else if (!pivot_is_valid)
    found_zero_pivot = true;

#if 0
      sign code
#endif

  A_INFO();
}
}

// _solve_impl
{
// dst = m_transpositions * rhs;
#if 0
	dst = m_transpositions * rhs;//swap
#endif
  v3X0 = v3B0;
  v3X1 = v3B1;
  v3X2 = v3B2;

  // matrixL().solveInPlace(dst);
  {
    // triangular_solver_unroller LoopIndex=0
    // triangular_solver_unroller LoopIndex=1
    v3X1 -= m3A10 * v3X0;
    // triangular_solver_unroller LoopIndex=1
    v3X2 -= m3A20 * v3X0 + m3A21 * v3X1;
    X_INFO();
  }

  // vecD(vectorD()); -> m3A00 m3A11 m3A22

  // for (Index i = 0; i < vecD.size(); ++i)
  {
    if (fabs(m3A00) > min_f32)
      v3X0 /= m3A00;
    else
      v3X0 = imm_0_f32;

    if (fabs(m3A11) > min_f32)
      v3X1 /= m3A11;
    else
      v3X1 = imm_0_f32;

    if (fabs(m3A22) > min_f32)
      v3X2 /= m3A22;
    else
      v3X2 = imm_0_f32;
    X_INFO();
  }

  // matrixU().solveInPlace(dst);
  {
    // triangular_solver_unroller LoopIndex=0
    // triangular_solver_unroller LoopIndex=1
    v3X1 -= m3A21 * v3X2;
    X_INFO();
    // triangular_solver_unroller LoopIndex=1
    v3X0 -= m3A10 * v3X1 + m3A20 * v3X2;
    X_INFO();
  }
}

OUT_INFO();

pv3X[0] = v3X0;
pv3X[1] = v3X1;
pv3X[2] = v3X2;
}

::Eigen::Vector3f ldlt_solve(const ::Eigen::Matrix3f &A,
                             const ::Eigen::Vector3f &B) {
  ::Eigen::Vector3f X;
  ldlt_3_simd4(&A(0, 0), &B(0, 0), &X(0, 0));
  return X;
}
#endif // #if CODE_ACC_Triangulate >= 3

void Triangulate_ACC_array(float *p_idp, const float *pM4_T01,
                           const float *pV3_v0, const float *pV3_v1, int size) {
#if CODE_ACC_Triangulate == 1
  number_t idp;
  Mat4 T01;
  Vec3 v0;
  Vec3 v1;
  memcpy(&T01(0, 0), pM4_T01, sizeof(T01));
  memcpy(&v0(0, 0), pV3_v0, sizeof(v0));
  for (int idx = 0; idx < size; idx++) {
    idp = p_idp[idx];
    memcpy(&v1(0, 0), &pV3_v1[3 * idx], sizeof(v1));
#if 0
	bool rtn = MultiCameraEpipolarSearch_body.Triangulate(idp, T01, v0, v1);
#else
    bool rtn;
    do {
      Mat63 A = Mat63::Zero();
      Vec6 b = Vec6::Zero();
      A.block<3, 3>(0, 0) = Skew(v0);
      A.block<3, 3>(3, 0) = Skew(v1) * T01.block<3, 3>(0, 0).transpose();
      b.segment<3>(3) =
          Skew(v1) * T01.block<3, 3>(0, 0).transpose() * T01.block<3, 1>(0, 3);
      Vec3 s = A.transpose() * b;
      Mat3 AA = A.transpose() * A;
      Vec3 xyz = AA.ldlt().solve(s);

      if (xyz.z() < 0) {
        rtn = false;
        break;
      }
      idp = 1.0 / xyz.norm();

      Vec3 v1_0 = T01.block<3, 3>(0, 0) * v1;
      number_t cos_theta = v1_0.dot(v0);
      //  if (cos_theta > 0.99999) {
      //    return false;
      //  }

      rtn = true;
      break;
    } while (0);
#endif
    if (rtn == false) {
      idp = -1.0f;
    }
    p_idp[idx] = idp;
  }
#endif
#if (CODE_ACC_Triangulate == 2) || (CODE_ACC_Triangulate == 3)
  Mat3 A1 = Mat3::Zero();
  A1(0, 1) = -pV3_v0[2];
  A1(0, 2) = pV3_v0[1];
  A1(1, 0) = pV3_v0[2];
  A1(1, 2) = -pV3_v0[0];
  A1(2, 0) = -pV3_v0[1];
  A1(2, 1) = pV3_v0[0];
  Mat3 AA1 = A1.transpose() * A1;

  float32x4x4_t T01_f32x4x3 = vld4q_f32(pM4_T01);
  float32x4_t v0_f32x4 = vld1q_f32(pV3_v0);

  if (size != 1) {
    for (int index = 0; index < size; index += 4) {
      float32x4x3_t v1_f32x4x3 = vld3q_f32(pV3_v1);
      pV3_v1 += 3 * 4;
      float32x4_t v1_0_f32x4 = v1_f32x4x3.val[0];
      float32x4_t v1_1_f32x4 = v1_f32x4x3.val[1];
      float32x4_t v1_2_f32x4 = v1_f32x4x3.val[2];

      float32x4_t A2_00_f32x4x3 = T01_f32x4x3.val[0][2] * v1_1_f32x4 -
                                  T01_f32x4x3.val[0][1] * v1_2_f32x4;
      float32x4_t A2_01_f32x4x3 = T01_f32x4x3.val[1][2] * v1_1_f32x4 -
                                  T01_f32x4x3.val[1][1] * v1_2_f32x4;
      float32x4_t A2_02_f32x4x3 = T01_f32x4x3.val[2][2] * v1_1_f32x4 -
                                  T01_f32x4x3.val[2][1] * v1_2_f32x4;
      float32x4_t A2_10_f32x4x3 = -T01_f32x4x3.val[0][2] * v1_0_f32x4 +
                                  T01_f32x4x3.val[0][0] * v1_2_f32x4;
      float32x4_t A2_11_f32x4x3 = -T01_f32x4x3.val[1][2] * v1_0_f32x4 +
                                  T01_f32x4x3.val[1][0] * v1_2_f32x4;
      float32x4_t A2_12_f32x4x3 = -T01_f32x4x3.val[2][2] * v1_0_f32x4 +
                                  T01_f32x4x3.val[2][0] * v1_2_f32x4;
      float32x4_t A2_20_f32x4x3 = T01_f32x4x3.val[0][1] * v1_0_f32x4 -
                                  T01_f32x4x3.val[0][0] * v1_1_f32x4;
      float32x4_t A2_21_f32x4x3 = T01_f32x4x3.val[1][1] * v1_0_f32x4 -
                                  T01_f32x4x3.val[1][0] * v1_1_f32x4;
      float32x4_t A2_22_f32x4x3 = T01_f32x4x3.val[2][1] * v1_0_f32x4 -
                                  T01_f32x4x3.val[2][0] * v1_1_f32x4;

      float32x4_t b_0_f32x4x3 = A2_02_f32x4x3 * T01_f32x4x3.val[2][3] +
                                A2_01_f32x4x3 * T01_f32x4x3.val[1][3] +
                                A2_00_f32x4x3 * T01_f32x4x3.val[0][3];
      float32x4_t b_1_f32x4x3 = A2_12_f32x4x3 * T01_f32x4x3.val[2][3] +
                                A2_11_f32x4x3 * T01_f32x4x3.val[1][3] +
                                A2_10_f32x4x3 * T01_f32x4x3.val[0][3];
      float32x4_t b_2_f32x4x3 = A2_22_f32x4x3 * T01_f32x4x3.val[2][3] +
                                A2_21_f32x4x3 * T01_f32x4x3.val[1][3] +
                                A2_20_f32x4x3 * T01_f32x4x3.val[0][3];

      float32x4_t s_0_f32x4x3 = A2_20_f32x4x3 * b_2_f32x4x3 +
                                A2_10_f32x4x3 * b_1_f32x4x3 +
                                A2_00_f32x4x3 * b_0_f32x4x3;
      float32x4_t s_1_f32x4x3 = A2_21_f32x4x3 * b_2_f32x4x3 +
                                A2_11_f32x4x3 * b_1_f32x4x3 +
                                A2_01_f32x4x3 * b_0_f32x4x3;
      float32x4_t s_2_f32x4x3 = A2_22_f32x4x3 * b_2_f32x4x3 +
                                A2_12_f32x4x3 * b_1_f32x4x3 +
                                A2_02_f32x4x3 * b_0_f32x4x3;

      float32x4_t AA2_00_f32x4x3 = A2_20_f32x4x3 * A2_20_f32x4x3 +
                                   A2_10_f32x4x3 * A2_10_f32x4x3 +
                                   A2_00_f32x4x3 * A2_00_f32x4x3;
      float32x4_t AA2_01_f32x4x3 = A2_20_f32x4x3 * A2_21_f32x4x3 +
                                   A2_10_f32x4x3 * A2_11_f32x4x3 +
                                   A2_00_f32x4x3 * A2_01_f32x4x3;
      float32x4_t AA2_02_f32x4x3 = A2_20_f32x4x3 * A2_22_f32x4x3 +
                                   A2_10_f32x4x3 * A2_12_f32x4x3 +
                                   A2_00_f32x4x3 * A2_02_f32x4x3;
      float32x4_t AA2_11_f32x4x3 = A2_21_f32x4x3 * A2_21_f32x4x3 +
                                   A2_11_f32x4x3 * A2_11_f32x4x3 +
                                   A2_01_f32x4x3 * A2_01_f32x4x3;
      float32x4_t AA2_12_f32x4x3 = A2_21_f32x4x3 * A2_22_f32x4x3 +
                                   A2_11_f32x4x3 * A2_12_f32x4x3 +
                                   A2_01_f32x4x3 * A2_02_f32x4x3;
      float32x4_t AA2_22_f32x4x3 = A2_22_f32x4x3 * A2_22_f32x4x3 +
                                   A2_12_f32x4x3 * A2_12_f32x4x3 +
                                   A2_02_f32x4x3 * A2_02_f32x4x3;

#if 0
#elif 1
      float32x4_t AA_00_f32x4x3 =
          A2_20_f32x4x3 * A2_20_f32x4x3 + A2_10_f32x4x3 * A2_10_f32x4x3 +
          A2_00_f32x4x3 * A2_00_f32x4x3 + vdupq_n_f32(AA1(0, 0));
      float32x4_t AA_01_f32x4x3 =
          A2_20_f32x4x3 * A2_21_f32x4x3 + A2_10_f32x4x3 * A2_11_f32x4x3 +
          A2_00_f32x4x3 * A2_01_f32x4x3 + vdupq_n_f32(AA1(0, 1));
      float32x4_t AA_02_f32x4x3 =
          A2_20_f32x4x3 * A2_22_f32x4x3 + A2_10_f32x4x3 * A2_12_f32x4x3 +
          A2_00_f32x4x3 * A2_02_f32x4x3 + vdupq_n_f32(AA1(0, 2));
      float32x4_t AA_11_f32x4x3 =
          A2_21_f32x4x3 * A2_21_f32x4x3 + A2_11_f32x4x3 * A2_11_f32x4x3 +
          A2_01_f32x4x3 * A2_01_f32x4x3 + vdupq_n_f32(AA1(1, 1));
      float32x4_t AA_12_f32x4x3 =
          A2_21_f32x4x3 * A2_22_f32x4x3 + A2_11_f32x4x3 * A2_12_f32x4x3 +
          A2_01_f32x4x3 * A2_02_f32x4x3 + vdupq_n_f32(AA1(1, 2));
      float32x4_t AA_22_f32x4x3 =
          A2_22_f32x4x3 * A2_22_f32x4x3 + A2_12_f32x4x3 * A2_12_f32x4x3 +
          A2_02_f32x4x3 * A2_02_f32x4x3 + vdupq_n_f32(AA1(2, 2));
#elif 1
      float32x4_t AA_00_f32x4x3 =
          A2_20_f32x4x3 * A2_20_f32x4x3 + A2_10_f32x4x3 * A2_10_f32x4x3 +
          A2_00_f32x4x3 * A2_00_f32x4x3 + vdupq_n_f32(AA1(0, 0));
      float32x4_t AA_01_f32x4x3 =
          A2_20_f32x4x3 * A2_21_f32x4x3 + A2_10_f32x4x3 * A2_11_f32x4x3 +
          A2_00_f32x4x3 * A2_01_f32x4x3 + vdupq_n_f32(AA1(0, 1));
      float32x4_t AA_02_f32x4x3 =
          A2_20_f32x4x3 * A2_22_f32x4x3 + A2_10_f32x4x3 * A2_12_f32x4x3 +
          A2_00_f32x4x3 * A2_02_f32x4x3 + vdupq_n_f32(AA1(0, 2));
      float32x4_t AA_11_f32x4x3 =
          A2_21_f32x4x3 * A2_21_f32x4x3 + A2_11_f32x4x3 * A2_11_f32x4x3 +
          A2_01_f32x4x3 * A2_01_f32x4x3 + vdupq_n_f32(AA1(1, 1));
      float32x4_t AA_12_f32x4x3 =
          A2_21_f32x4x3 * A2_22_f32x4x3 + A2_11_f32x4x3 * A2_12_f32x4x3 +
          A2_01_f32x4x3 * A2_02_f32x4x3 + vdupq_n_f32(AA1(1, 2));
      float32x4_t AA_22_f32x4x3 =
          A2_22_f32x4x3 * A2_22_f32x4x3 + A2_12_f32x4x3 * A2_12_f32x4x3 +
          A2_02_f32x4x3 * A2_02_f32x4x3 + vdupq_n_f32(AA1(2, 2));
#elif 1
      float32x4_t AA_00_f32x4x3 = AA2_00_f32x4x3 + vdupq_n_f32(AA1(0, 0));
      float32x4_t AA_01_f32x4x3 = AA2_01_f32x4x3 + vdupq_n_f32(AA1(0, 1));
      float32x4_t AA_02_f32x4x3 = AA2_02_f32x4x3 + vdupq_n_f32(AA1(0, 2));
      float32x4_t AA_11_f32x4x3 = AA2_11_f32x4x3 + vdupq_n_f32(AA1(1, 1));
      float32x4_t AA_12_f32x4x3 = AA2_12_f32x4x3 + vdupq_n_f32(AA1(1, 2));
      float32x4_t AA_22_f32x4x3 = AA2_22_f32x4x3 + vdupq_n_f32(AA1(2, 2));
#endif

      Mat3 AA[4];
      Vec3 s[4];
      Vec3 xyz[4];
      AA[0](0, 0) = vgetq_lane_f32(AA_00_f32x4x3, 0);
      AA[0](0, 1) = vgetq_lane_f32(AA_01_f32x4x3, 0);
      AA[0](0, 2) = vgetq_lane_f32(AA_02_f32x4x3, 0);
      AA[0](1, 0) = vgetq_lane_f32(AA_01_f32x4x3, 0);
      AA[0](1, 1) = vgetq_lane_f32(AA_11_f32x4x3, 0);
      AA[0](1, 2) = vgetq_lane_f32(AA_12_f32x4x3, 0);
      AA[0](2, 0) = vgetq_lane_f32(AA_02_f32x4x3, 0);
      AA[0](2, 1) = vgetq_lane_f32(AA_12_f32x4x3, 0);
      AA[0](2, 2) = vgetq_lane_f32(AA_22_f32x4x3, 0);
      s[0][0] = vgetq_lane_f32(s_0_f32x4x3, 0);
      s[0][1] = vgetq_lane_f32(s_1_f32x4x3, 0);
      s[0][2] = vgetq_lane_f32(s_2_f32x4x3, 0);

      AA[1](0, 0) = vgetq_lane_f32(AA_00_f32x4x3, 1);
      AA[1](0, 1) = vgetq_lane_f32(AA_01_f32x4x3, 1);
      AA[1](0, 2) = vgetq_lane_f32(AA_02_f32x4x3, 1);
      AA[1](1, 0) = vgetq_lane_f32(AA_01_f32x4x3, 1);
      AA[1](1, 1) = vgetq_lane_f32(AA_11_f32x4x3, 1);
      AA[1](1, 2) = vgetq_lane_f32(AA_12_f32x4x3, 1);
      AA[1](2, 0) = vgetq_lane_f32(AA_02_f32x4x3, 1);
      AA[1](2, 1) = vgetq_lane_f32(AA_12_f32x4x3, 1);
      AA[1](2, 2) = vgetq_lane_f32(AA_22_f32x4x3, 1);
      s[1][0] = vgetq_lane_f32(s_0_f32x4x3, 1);
      s[1][1] = vgetq_lane_f32(s_1_f32x4x3, 1);
      s[1][2] = vgetq_lane_f32(s_2_f32x4x3, 1);

      AA[2](0, 0) = vgetq_lane_f32(AA_00_f32x4x3, 2);
      AA[2](0, 1) = vgetq_lane_f32(AA_01_f32x4x3, 2);
      AA[2](0, 2) = vgetq_lane_f32(AA_02_f32x4x3, 2);
      AA[2](1, 0) = vgetq_lane_f32(AA_01_f32x4x3, 2);
      AA[2](1, 1) = vgetq_lane_f32(AA_11_f32x4x3, 2);
      AA[2](1, 2) = vgetq_lane_f32(AA_12_f32x4x3, 2);
      AA[2](2, 0) = vgetq_lane_f32(AA_02_f32x4x3, 2);
      AA[2](2, 1) = vgetq_lane_f32(AA_12_f32x4x3, 2);
      AA[2](2, 2) = vgetq_lane_f32(AA_22_f32x4x3, 2);
      s[2][0] = vgetq_lane_f32(s_0_f32x4x3, 2);
      s[2][1] = vgetq_lane_f32(s_1_f32x4x3, 2);
      s[2][2] = vgetq_lane_f32(s_2_f32x4x3, 2);

      AA[3](0, 0) = vgetq_lane_f32(AA_00_f32x4x3, 3);
      AA[3](0, 1) = vgetq_lane_f32(AA_01_f32x4x3, 3);
      AA[3](0, 2) = vgetq_lane_f32(AA_02_f32x4x3, 3);
      AA[3](1, 0) = vgetq_lane_f32(AA_01_f32x4x3, 3);
      AA[3](1, 1) = vgetq_lane_f32(AA_11_f32x4x3, 3);
      AA[3](1, 2) = vgetq_lane_f32(AA_12_f32x4x3, 3);
      AA[3](2, 0) = vgetq_lane_f32(AA_02_f32x4x3, 3);
      AA[3](2, 1) = vgetq_lane_f32(AA_12_f32x4x3, 3);
      AA[3](2, 2) = vgetq_lane_f32(AA_22_f32x4x3, 3);
      s[3][0] = vgetq_lane_f32(s_0_f32x4x3, 3);
      s[3][1] = vgetq_lane_f32(s_1_f32x4x3, 3);
      s[3][2] = vgetq_lane_f32(s_2_f32x4x3, 3);

#if CODE_ACC_Triangulate == 3
      xyz[0] = ldlt_solve(AA[0], s[0]);
      xyz[1] = ldlt_solve(AA[1], s[1]);
      xyz[2] = ldlt_solve(AA[2], s[2]);
      xyz[3] = ldlt_solve(AA[3], s[3]);
#else
      xyz[0] = AA[0].ldlt().solve(s[0]);
      xyz[1] = AA[1].ldlt().solve(s[1]);
      xyz[2] = AA[2].ldlt().solve(s[2]);
      xyz[3] = AA[3].ldlt().solve(s[3]);
#endif

#if 0
#define std_cout(_var)                                                         \
  do {                                                                         \
    std::cout << #_var << std::endl << _var << std::endl;                      \
  } while (0)
std_cout(AA[0]);
std_cout(s[0]);
std_cout(xyz[0]);

{static int Cnt = 0; if(Cnt++ == 10) {
exit(1);
}}
#endif

      float32x4x3_t xyz_f32x4x3 = vld3q_f32(&xyz[0](0, 0));

      float32x4_t xyz_0_f32x4x3 = xyz_f32x4x3.val[0];
      float32x4_t xyz_1_f32x4x3 = xyz_f32x4x3.val[1];
      float32x4_t xyz_2_f32x4x3 = xyz_f32x4x3.val[2];

      float32x4_t squaredNorm_f32x4x3 = xyz_2_f32x4x3 * xyz_2_f32x4x3 +
                                        xyz_1_f32x4x3 * xyz_1_f32x4x3 +
                                        xyz_0_f32x4x3 * xyz_0_f32x4x3;
      float32x4_t norm_f32x4x3 = vsqrtq_f32(squaredNorm_f32x4x3);
      float32x4_t idp_f32x4x3 = vdupq_n_f32(1.0f) / norm_f32x4x3;

      float32x4_t v1_0_0_f32x4 = T01_f32x4x3.val[0][2] * v1_2_f32x4 +
                                 T01_f32x4x3.val[0][1] * v1_1_f32x4 +
                                 T01_f32x4x3.val[0][0] * v1_0_f32x4;
      float32x4_t v1_0_1_f32x4 = T01_f32x4x3.val[1][2] * v1_2_f32x4 +
                                 T01_f32x4x3.val[1][1] * v1_1_f32x4 +
                                 T01_f32x4x3.val[1][0] * v1_0_f32x4;
      float32x4_t v1_0_2_f32x4 = T01_f32x4x3.val[2][2] * v1_2_f32x4 +
                                 T01_f32x4x3.val[2][1] * v1_1_f32x4 +
                                 T01_f32x4x3.val[2][0] * v1_0_f32x4;
      float32x4_t cos_theta_f32x4 = v1_0_2_f32x4 * v0_f32x4[2] +
                                    v1_0_1_f32x4 * v0_f32x4[1] +
                                    v1_0_0_f32x4 * v0_f32x4[0];

      uint32x4_t valid_u32x4 = (xyz_2_f32x4x3 >= vdupq_n_f32(0)) &&
                               (cos_theta_f32x4 <= vdupq_n_f32(0.99999));
      idp_f32x4x3 = vbslq_f32(valid_u32x4, idp_f32x4x3, vdupq_n_f32(-1));

      vst1q_f32(p_idp, idp_f32x4x3);
      p_idp += 4;
    }
  } else {
    float32x4x3_t v1_f32x4x3 = vld3q_f32(pV3_v1);
    float32x4_t v1_0_f32x4 = v1_f32x4x3.val[0];
    float32x4_t v1_1_f32x4 = v1_f32x4x3.val[1];
    float32x4_t v1_2_f32x4 = v1_f32x4x3.val[2];

    float32x4_t A2_00_f32x4x3 =
        T01_f32x4x3.val[0][2] * v1_1_f32x4 - T01_f32x4x3.val[0][1] * v1_2_f32x4;
    float32x4_t A2_01_f32x4x3 =
        T01_f32x4x3.val[1][2] * v1_1_f32x4 - T01_f32x4x3.val[1][1] * v1_2_f32x4;
    float32x4_t A2_02_f32x4x3 =
        T01_f32x4x3.val[2][2] * v1_1_f32x4 - T01_f32x4x3.val[2][1] * v1_2_f32x4;
    float32x4_t A2_10_f32x4x3 = -T01_f32x4x3.val[0][2] * v1_0_f32x4 +
                                T01_f32x4x3.val[0][0] * v1_2_f32x4;
    float32x4_t A2_11_f32x4x3 = -T01_f32x4x3.val[1][2] * v1_0_f32x4 +
                                T01_f32x4x3.val[1][0] * v1_2_f32x4;
    float32x4_t A2_12_f32x4x3 = -T01_f32x4x3.val[2][2] * v1_0_f32x4 +
                                T01_f32x4x3.val[2][0] * v1_2_f32x4;
    float32x4_t A2_20_f32x4x3 =
        T01_f32x4x3.val[0][1] * v1_0_f32x4 - T01_f32x4x3.val[0][0] * v1_1_f32x4;
    float32x4_t A2_21_f32x4x3 =
        T01_f32x4x3.val[1][1] * v1_0_f32x4 - T01_f32x4x3.val[1][0] * v1_1_f32x4;
    float32x4_t A2_22_f32x4x3 =
        T01_f32x4x3.val[2][1] * v1_0_f32x4 - T01_f32x4x3.val[2][0] * v1_1_f32x4;

    float32x4_t b_0_f32x4x3 = A2_02_f32x4x3 * T01_f32x4x3.val[2][3] +
                              A2_01_f32x4x3 * T01_f32x4x3.val[1][3] +
                              A2_00_f32x4x3 * T01_f32x4x3.val[0][3];
    float32x4_t b_1_f32x4x3 = A2_12_f32x4x3 * T01_f32x4x3.val[2][3] +
                              A2_11_f32x4x3 * T01_f32x4x3.val[1][3] +
                              A2_10_f32x4x3 * T01_f32x4x3.val[0][3];
    float32x4_t b_2_f32x4x3 = A2_22_f32x4x3 * T01_f32x4x3.val[2][3] +
                              A2_21_f32x4x3 * T01_f32x4x3.val[1][3] +
                              A2_20_f32x4x3 * T01_f32x4x3.val[0][3];

    float32x4_t s_0_f32x4x3 = A2_20_f32x4x3 * b_2_f32x4x3 +
                              A2_10_f32x4x3 * b_1_f32x4x3 +
                              A2_00_f32x4x3 * b_0_f32x4x3;
    float32x4_t s_1_f32x4x3 = A2_21_f32x4x3 * b_2_f32x4x3 +
                              A2_11_f32x4x3 * b_1_f32x4x3 +
                              A2_01_f32x4x3 * b_0_f32x4x3;
    float32x4_t s_2_f32x4x3 = A2_22_f32x4x3 * b_2_f32x4x3 +
                              A2_12_f32x4x3 * b_1_f32x4x3 +
                              A2_02_f32x4x3 * b_0_f32x4x3;

    float32x4_t AA2_00_f32x4x3 = A2_20_f32x4x3 * A2_20_f32x4x3 +
                                 A2_10_f32x4x3 * A2_10_f32x4x3 +
                                 A2_00_f32x4x3 * A2_00_f32x4x3;
    float32x4_t AA2_01_f32x4x3 = A2_20_f32x4x3 * A2_21_f32x4x3 +
                                 A2_10_f32x4x3 * A2_11_f32x4x3 +
                                 A2_00_f32x4x3 * A2_01_f32x4x3;
    float32x4_t AA2_02_f32x4x3 = A2_20_f32x4x3 * A2_22_f32x4x3 +
                                 A2_10_f32x4x3 * A2_12_f32x4x3 +
                                 A2_00_f32x4x3 * A2_02_f32x4x3;
    float32x4_t AA2_11_f32x4x3 = A2_21_f32x4x3 * A2_21_f32x4x3 +
                                 A2_11_f32x4x3 * A2_11_f32x4x3 +
                                 A2_01_f32x4x3 * A2_01_f32x4x3;
    float32x4_t AA2_12_f32x4x3 = A2_21_f32x4x3 * A2_22_f32x4x3 +
                                 A2_11_f32x4x3 * A2_12_f32x4x3 +
                                 A2_01_f32x4x3 * A2_02_f32x4x3;
    float32x4_t AA2_22_f32x4x3 = A2_22_f32x4x3 * A2_22_f32x4x3 +
                                 A2_12_f32x4x3 * A2_12_f32x4x3 +
                                 A2_02_f32x4x3 * A2_02_f32x4x3;

#if 0
#elif 1
    float32x4_t AA_00_f32x4x3 =
        A2_20_f32x4x3 * A2_20_f32x4x3 + A2_10_f32x4x3 * A2_10_f32x4x3 +
        A2_00_f32x4x3 * A2_00_f32x4x3 + vdupq_n_f32(AA1(0, 0));
    float32x4_t AA_01_f32x4x3 =
        A2_20_f32x4x3 * A2_21_f32x4x3 + A2_10_f32x4x3 * A2_11_f32x4x3 +
        A2_00_f32x4x3 * A2_01_f32x4x3 + vdupq_n_f32(AA1(0, 1));
    float32x4_t AA_02_f32x4x3 =
        A2_20_f32x4x3 * A2_22_f32x4x3 + A2_10_f32x4x3 * A2_12_f32x4x3 +
        A2_00_f32x4x3 * A2_02_f32x4x3 + vdupq_n_f32(AA1(0, 2));
    float32x4_t AA_11_f32x4x3 =
        A2_21_f32x4x3 * A2_21_f32x4x3 + A2_11_f32x4x3 * A2_11_f32x4x3 +
        A2_01_f32x4x3 * A2_01_f32x4x3 + vdupq_n_f32(AA1(1, 1));
    float32x4_t AA_12_f32x4x3 =
        A2_21_f32x4x3 * A2_22_f32x4x3 + A2_11_f32x4x3 * A2_12_f32x4x3 +
        A2_01_f32x4x3 * A2_02_f32x4x3 + vdupq_n_f32(AA1(1, 2));
    float32x4_t AA_22_f32x4x3 =
        A2_22_f32x4x3 * A2_22_f32x4x3 + A2_12_f32x4x3 * A2_12_f32x4x3 +
        A2_02_f32x4x3 * A2_02_f32x4x3 + vdupq_n_f32(AA1(2, 2));
#elif 1
    float32x4_t AA_00_f32x4x3 =
        A2_20_f32x4x3 * A2_20_f32x4x3 + A2_10_f32x4x3 * A2_10_f32x4x3 +
        A2_00_f32x4x3 * A2_00_f32x4x3 + vdupq_n_f32(AA1(0, 0));
    float32x4_t AA_01_f32x4x3 =
        A2_20_f32x4x3 * A2_21_f32x4x3 + A2_10_f32x4x3 * A2_11_f32x4x3 +
        A2_00_f32x4x3 * A2_01_f32x4x3 + vdupq_n_f32(AA1(0, 1));
    float32x4_t AA_02_f32x4x3 =
        A2_20_f32x4x3 * A2_22_f32x4x3 + A2_10_f32x4x3 * A2_12_f32x4x3 +
        A2_00_f32x4x3 * A2_02_f32x4x3 + vdupq_n_f32(AA1(0, 2));
    float32x4_t AA_11_f32x4x3 =
        A2_21_f32x4x3 * A2_21_f32x4x3 + A2_11_f32x4x3 * A2_11_f32x4x3 +
        A2_01_f32x4x3 * A2_01_f32x4x3 + vdupq_n_f32(AA1(1, 1));
    float32x4_t AA_12_f32x4x3 =
        A2_21_f32x4x3 * A2_22_f32x4x3 + A2_11_f32x4x3 * A2_12_f32x4x3 +
        A2_01_f32x4x3 * A2_02_f32x4x3 + vdupq_n_f32(AA1(1, 2));
    float32x4_t AA_22_f32x4x3 =
        A2_22_f32x4x3 * A2_22_f32x4x3 + A2_12_f32x4x3 * A2_12_f32x4x3 +
        A2_02_f32x4x3 * A2_02_f32x4x3 + vdupq_n_f32(AA1(2, 2));
#elif 1
    float32x4_t AA_00_f32x4x3 = AA2_00_f32x4x3 + vdupq_n_f32(AA1(0, 0));
    float32x4_t AA_01_f32x4x3 = AA2_01_f32x4x3 + vdupq_n_f32(AA1(0, 1));
    float32x4_t AA_02_f32x4x3 = AA2_02_f32x4x3 + vdupq_n_f32(AA1(0, 2));
    float32x4_t AA_11_f32x4x3 = AA2_11_f32x4x3 + vdupq_n_f32(AA1(1, 1));
    float32x4_t AA_12_f32x4x3 = AA2_12_f32x4x3 + vdupq_n_f32(AA1(1, 2));
    float32x4_t AA_22_f32x4x3 = AA2_22_f32x4x3 + vdupq_n_f32(AA1(2, 2));
#endif

    Mat3 AA[4];
    Vec3 s[4];
    Vec3 xyz[4];
    AA[0](0, 0) = vgetq_lane_f32(AA_00_f32x4x3, 0);
    AA[0](0, 1) = vgetq_lane_f32(AA_01_f32x4x3, 0);
    AA[0](0, 2) = vgetq_lane_f32(AA_02_f32x4x3, 0);
    AA[0](1, 0) = vgetq_lane_f32(AA_01_f32x4x3, 0);
    AA[0](1, 1) = vgetq_lane_f32(AA_11_f32x4x3, 0);
    AA[0](1, 2) = vgetq_lane_f32(AA_12_f32x4x3, 0);
    AA[0](2, 0) = vgetq_lane_f32(AA_02_f32x4x3, 0);
    AA[0](2, 1) = vgetq_lane_f32(AA_12_f32x4x3, 0);
    AA[0](2, 2) = vgetq_lane_f32(AA_22_f32x4x3, 0);
    s[0][0] = vgetq_lane_f32(s_0_f32x4x3, 0);
    s[0][1] = vgetq_lane_f32(s_1_f32x4x3, 0);
    s[0][2] = vgetq_lane_f32(s_2_f32x4x3, 0);

    AA[1](0, 0) = vgetq_lane_f32(AA_00_f32x4x3, 1);
    AA[1](0, 1) = vgetq_lane_f32(AA_01_f32x4x3, 1);
    AA[1](0, 2) = vgetq_lane_f32(AA_02_f32x4x3, 1);
    AA[1](1, 0) = vgetq_lane_f32(AA_01_f32x4x3, 1);
    AA[1](1, 1) = vgetq_lane_f32(AA_11_f32x4x3, 1);
    AA[1](1, 2) = vgetq_lane_f32(AA_12_f32x4x3, 1);
    AA[1](2, 0) = vgetq_lane_f32(AA_02_f32x4x3, 1);
    AA[1](2, 1) = vgetq_lane_f32(AA_12_f32x4x3, 1);
    AA[1](2, 2) = vgetq_lane_f32(AA_22_f32x4x3, 1);
    s[1][0] = vgetq_lane_f32(s_0_f32x4x3, 1);
    s[1][1] = vgetq_lane_f32(s_1_f32x4x3, 1);
    s[1][2] = vgetq_lane_f32(s_2_f32x4x3, 1);

    AA[2](0, 0) = vgetq_lane_f32(AA_00_f32x4x3, 2);
    AA[2](0, 1) = vgetq_lane_f32(AA_01_f32x4x3, 2);
    AA[2](0, 2) = vgetq_lane_f32(AA_02_f32x4x3, 2);
    AA[2](1, 0) = vgetq_lane_f32(AA_01_f32x4x3, 2);
    AA[2](1, 1) = vgetq_lane_f32(AA_11_f32x4x3, 2);
    AA[2](1, 2) = vgetq_lane_f32(AA_12_f32x4x3, 2);
    AA[2](2, 0) = vgetq_lane_f32(AA_02_f32x4x3, 2);
    AA[2](2, 1) = vgetq_lane_f32(AA_12_f32x4x3, 2);
    AA[2](2, 2) = vgetq_lane_f32(AA_22_f32x4x3, 2);
    s[2][0] = vgetq_lane_f32(s_0_f32x4x3, 2);
    s[2][1] = vgetq_lane_f32(s_1_f32x4x3, 2);
    s[2][2] = vgetq_lane_f32(s_2_f32x4x3, 2);

    AA[3](0, 0) = vgetq_lane_f32(AA_00_f32x4x3, 3);
    AA[3](0, 1) = vgetq_lane_f32(AA_01_f32x4x3, 3);
    AA[3](0, 2) = vgetq_lane_f32(AA_02_f32x4x3, 3);
    AA[3](1, 0) = vgetq_lane_f32(AA_01_f32x4x3, 3);
    AA[3](1, 1) = vgetq_lane_f32(AA_11_f32x4x3, 3);
    AA[3](1, 2) = vgetq_lane_f32(AA_12_f32x4x3, 3);
    AA[3](2, 0) = vgetq_lane_f32(AA_02_f32x4x3, 3);
    AA[3](2, 1) = vgetq_lane_f32(AA_12_f32x4x3, 3);
    AA[3](2, 2) = vgetq_lane_f32(AA_22_f32x4x3, 3);
    s[3][0] = vgetq_lane_f32(s_0_f32x4x3, 3);
    s[3][1] = vgetq_lane_f32(s_1_f32x4x3, 3);
    s[3][2] = vgetq_lane_f32(s_2_f32x4x3, 3);

    xyz[0] = AA[0].ldlt().solve(s[0]);
    xyz[1] = AA[1].ldlt().solve(s[1]);
    xyz[2] = AA[2].ldlt().solve(s[2]);
    xyz[3] = AA[3].ldlt().solve(s[3]);

    float32x4x3_t xyz_f32x4x3 = vld3q_f32(&xyz[0](0, 0));

    float32x4_t xyz_0_f32x4x3 = xyz_f32x4x3.val[0];
    float32x4_t xyz_1_f32x4x3 = xyz_f32x4x3.val[1];
    float32x4_t xyz_2_f32x4x3 = xyz_f32x4x3.val[2];

    float32x4_t squaredNorm_f32x4x3 = xyz_2_f32x4x3 * xyz_2_f32x4x3 +
                                      xyz_1_f32x4x3 * xyz_1_f32x4x3 +
                                      xyz_0_f32x4x3 * xyz_0_f32x4x3;
    float32x4_t norm_f32x4x3 = vsqrtq_f32(squaredNorm_f32x4x3);
    float32x4_t idp_f32x4x3 = vdupq_n_f32(1.0f) / norm_f32x4x3;

    float32x4_t v1_0_0_f32x4 = T01_f32x4x3.val[0][2] * v1_2_f32x4 +
                               T01_f32x4x3.val[0][1] * v1_1_f32x4 +
                               T01_f32x4x3.val[0][0] * v1_0_f32x4;
    float32x4_t v1_0_1_f32x4 = T01_f32x4x3.val[1][2] * v1_2_f32x4 +
                               T01_f32x4x3.val[1][1] * v1_1_f32x4 +
                               T01_f32x4x3.val[1][0] * v1_0_f32x4;
    float32x4_t v1_0_2_f32x4 = T01_f32x4x3.val[2][2] * v1_2_f32x4 +
                               T01_f32x4x3.val[2][1] * v1_1_f32x4 +
                               T01_f32x4x3.val[2][0] * v1_0_f32x4;
    float32x4_t cos_theta_f32x4 = v1_0_2_f32x4 * v0_f32x4[2] +
                                  v1_0_1_f32x4 * v0_f32x4[1] +
                                  v1_0_0_f32x4 * v0_f32x4[0];

    uint32x4_t valid_u32x4 = (xyz_2_f32x4x3 >= vdupq_n_f32(0)) &&
                             (cos_theta_f32x4 <= vdupq_n_f32(0.99999));
    idp_f32x4x3 = vbslq_f32(valid_u32x4, idp_f32x4x3, vdupq_n_f32(-1));

    *p_idp = vgetq_lane_f32(idp_f32x4x3, 0);
  }

#endif
}
#endif

} // namespace dso