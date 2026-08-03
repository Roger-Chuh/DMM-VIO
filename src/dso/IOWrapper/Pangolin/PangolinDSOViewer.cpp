/**
 * This file is part of DSO, written by Jakob Engel.
 * It has been modified by Lukas von Stumberg for the inclusion in DM-VIO
 * (http://vision.in.tum.de/dm-vio).
 *
 * Copyright 2022 Lukas von Stumberg <lukas dot stumberg at tum dot de>
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#include "PangolinDSOViewer.h"
#include "KeyFrameDisplay.h"

#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"
#include "util/globalCalib.h"
#include "util/settings.h"

namespace dso {
namespace IOWrap {

PangolinDSOViewer::PangolinDSOViewer(int w, int h, bool startRunThread,
                                     std::shared_ptr<dmvio::SettingsUtil> settingsUtilPassed,
                                     std::shared_ptr<double> normalizeCamSize, MultiCamera* p_multi_camera_)
    : HCalib(0),
      settingsUtil(std::move(settingsUtilPassed)),
      normalizeCamSize(normalizeCamSize),
      p_multi_camera(p_multi_camera_) {
  this->w = w;
  this->h = h;
  running = true;

  {
    boost::unique_lock<boost::mutex> lk(openImagesMutex);
    internalVideoImg = new MinimalImageB3(w, this->h);
    internalKFImg = new MinimalImageB3(w, this->h);
    internalResImg = new MinimalImageB3(w, this->h);
    videoImgChanged = kfImgChanged = resImgChanged = true;

    internalVideoImg->setBlack();
    internalKFImg->setBlack();
    internalResImg->setBlack();
  }

  {
    currentCam = new KeyFrameDisplay(p_multi_camera);
    currentGTCam = new KeyFrameDisplay(p_multi_camera);
  }

  needReset = false;

  if (startRunThread) runThread = boost::thread(&PangolinDSOViewer::run, this);
}

PangolinDSOViewer::~PangolinDSOViewer() {
  close();
  if (runThread.joinable()) runThread.join();
}

void PangolinDSOViewer::run() {
  printf("START PANGOLIN!\n");

  pangolin::CreateWindowAndBind("Main", 2 * w * kCameraNumUsed, 2 * h);
  const int UI_WIDTH = 180;
#ifdef USE_MULTI_CAM
  const int PointCloud_Start = 1 * UI_WIDTH;
#else
  const int PointCloud_Start = UI_WIDTH;
#endif
  glEnable(GL_DEPTH_TEST);

  // 3D visualization
  pangolin::OpenGlRenderState Visualization3D_camera(
      pangolin::ProjectionMatrix(w, h, 400, 400, w / 2, h / 2, 0.1, 1000),
      pangolin::ModelViewLookAt(-0, -5, -10, 0, 0, 0, pangolin::AxisNegY));

#ifdef USE_MULTI_CAM
  pangolin::View& Visualization3D_display =
      pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(PointCloud_Start), 1.0, -w / (float)h)
          .SetHandler(new pangolin::Handler3D(Visualization3D_camera));
#else
  pangolin::View& Visualization3D_display =
      pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -w / (float)h)
          .SetHandler(new pangolin::Handler3D(Visualization3D_camera));
#endif
  // 3 images
  pangolin::View& d_kfDepth = pangolin::Display("imgKFDepth").SetAspect(w / (float)(h * kCameraNumUsed));

  pangolin::View& d_video = pangolin::Display("imgVideo").SetAspect(w / (float)(h * kCameraNumUsed));

  pangolin::View& d_residual = pangolin::Display("imgResidual").SetAspect(w / (float)(h * kCameraNumUsed));

  pangolin::GlTexture texKFDepth(w, h * kCameraNumUsed, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
  pangolin::GlTexture texVideo(w, h * kCameraNumUsed, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
  pangolin::GlTexture texResidual(w, h * kCameraNumUsed, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

  float ratio = 0.3;
  if (kCameraNumUsed > 1) {
    ratio = 0.3 * kCameraNumUsed;
  }
#ifdef USE_MULTI_CAM
  pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
                 0.5)  // (float)(w * kCameraNumUsed) / (float)(h))
      .SetLayout(pangolin::LayoutEqual)
      .AddDisplay(d_kfDepth)
      .AddDisplay(d_video);
  //.AddDisplay(d_residual);
#else
  pangolin::CreateDisplay()
      .SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_WIDTH), 1.0)
      .SetLayout(pangolin::LayoutEqual)
      .AddDisplay(d_kfDepth)
      .AddDisplay(d_video)
      .AddDisplay(d_residual);
#endif

  // parameter reconfigure gui
  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

  pangolin::Var<int> settings_pointCloudMode("ui.PC_mode", 1, 1, 4, false);

  pangolin::Var<bool> settings_showKFCameras("ui.KFCam", false, true);
  pangolin::Var<bool> settings_showCurrentCamera("ui.CurrCam", true, true);
  pangolin::Var<bool> settings_showTrajectory("ui.Trajectory", false, true);
  pangolin::Var<bool> settings_showFullTrajectory("ui.FullTrajectory", true, true);
  pangolin::Var<bool> settings_showActiveConstraints("ui.ActiveConst", true, true);
  pangolin::Var<bool> settings_showAllConstraints("ui.AllConst", false, true);

  pangolin::Var<bool> settings_show3D("ui.show3D", true, true);
  pangolin::Var<bool> settings_showLiveDepth("ui.showDepth", true, true);
  pangolin::Var<bool> settings_showLiveVideo("ui.showVideo", true, true);
#ifndef USE_MULTI_CAM
  pangolin::Var<bool> settings_showLiveResidual("ui.showResidual", false, true);
#endif

  pangolin::Var<bool> settings_showFramesWindow("ui.showFramesWindow", false, true);
  pangolin::Var<bool> settings_showFullTracking("ui.showFullTracking", false, true);
  pangolin::Var<bool> settings_showCoarseTracking("ui.showCoarseTracking", false, true);

  pangolin::Var<int> settings_sparsity("ui.sparsity", 1, 1, 20, false);
  pangolin::Var<double> settings_scaledVarTH("ui.relVarTH", 0.001, 1e-10, 1e10, true);
  pangolin::Var<double> settings_absVarTH("ui.absVarTH", 0.001, 1e-10, 1e10, true);
  pangolin::Var<double> settings_minRelBS("ui.minRelativeBS", 0.1, 0, 1, false);

  pangolin::Var<bool> settings_resetButton("ui.Reset", false, false);

  pangolin::Var<int> settings_nPts("ui.activePoints", setting_desiredPointDensity, 50, 50000, false);
  pangolin::Var<int> settings_nCandidates("ui.pointCandidates", setting_desiredImmatureDensity, 50, 50000, false);
  pangolin::Var<int> settings_nMaxFrames("ui.maxFrames", setting_maxFrames, 7, 30, false);
  // pangolin::Var<int> settings_nMaxFrames("ui.maxFrames",setting_maxFrames,
  // 20,40, false);
  pangolin::Var<double> settings_kfFrequency("ui.kfFrequency", setting_kfGlobalWeight, 0.001, 3, false);
  pangolin::Var<double> settings_gradHistAdd("ui.minGradAdd", setting_minGradHistAdd, 0, 30, false);

  pangolin::Var<double> settings_trackFps("ui.Track fps", 0, 0, 0, false);
  pangolin::Var<double> settings_mapFps("ui.KF fps", 0, 0, 0, false);

  pangolin::Var<double> settings_Scale("ui.Scale", 1, 0, 0, false);
  pangolin::Var<std::string> setting_SystemStatus("ui.Status", "");

  if (settingsUtil) {
    settingsUtil->createPangolinSettings();
  }

  // Default hooks for exiting (Esc) and fullscreen (tab).
  while (running) {
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    // Clear entire screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (setting_render_display3D) {
      double sizeFactor = 1.0;

      // Activate efficiently by object
      Visualization3D_display.Activate(Visualization3D_camera);
      boost::unique_lock<boost::mutex> lk3d(model3DMutex);

      // Normalize cam size with scale.
      if (transformDSOToIMU && normalizeCamSize && *normalizeCamSize > 0) {
        sizeFactor = *normalizeCamSize / transformDSOToIMU->getScale();
      }

      // pangolin::glDrawColouredCube();
      int refreshed = 0;
      for (KeyFrameDisplay* fh : keyframes) {
        float blue[3] = {0, 0, 1};
        if (this->settings_showKFCameras) fh->drawCam(1, blue, 0.1 * sizeFactor);

        refreshed +=
            (int)(fh->refreshPC(refreshed < 10, this->settings_scaledVarTH, this->settings_absVarTH,
                                this->settings_pointCloudMode, this->settings_minRelBS, this->settings_sparsity));
        fh->drawPC(1);
      }
      if (this->settings_showCurrentCamera) currentCam->drawCam(2, 0, 0.2 * sizeFactor);

      float green[3] = {0, 1, 0};
      if (gtCamPoseSet) {
        currentGTCam->drawCam(2, green, 0.2 * sizeFactor);
      }

      drawConstraints();
      lk3d.unlock();
    }

    openImagesMutex.lock();
    if (videoImgChanged) texVideo.Upload(internalVideoImg->data, GL_BGR, GL_UNSIGNED_BYTE);
    if (kfImgChanged) texKFDepth.Upload(internalKFImg->data, GL_BGR, GL_UNSIGNED_BYTE);
    if (resImgChanged) texResidual.Upload(internalResImg->data, GL_BGR, GL_UNSIGNED_BYTE);
    videoImgChanged = kfImgChanged = resImgChanged = false;
    openImagesMutex.unlock();

    // update fps counters
    {
      openImagesMutex.lock();
      float sd = 0;
      for (float d : lastNMappingMs) sd += d;
      settings_mapFps = lastNMappingMs.size() * 1000.0f / sd;
      openImagesMutex.unlock();
    }
    {
      model3DMutex.lock();
      float sd = 0;
      for (float d : lastNTrackingMs) sd += d;
      settings_trackFps = lastNTrackingMs.size() * 1000.0f / sd;
      model3DMutex.unlock();
    }

    // Update scale and status text.
    {
      boost::unique_lock<boost::mutex> lk(model3DMutex);
      if (transformDSOToIMU) {
        settings_Scale = transformDSOToIMU->getScale();
      }
      switch (systemStatus) {
        case dmvio::VISUAL_INIT:
          setting_SystemStatus = "Visual-init";
          break;
        case dmvio::VISUAL_ONLY:
          setting_SystemStatus = "Visual-only";
          break;
        case dmvio::VISUAL_INERTIAL:
          setting_SystemStatus = "VIO";
          break;
      }
    }

    if (setting_render_displayVideo) {
      d_video.Activate();
      glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
      texVideo.RenderToViewportFlipY();
    }

    if (setting_render_displayDepth) {
      d_kfDepth.Activate();
      glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
      texKFDepth.RenderToViewportFlipY();
    }

    if (setting_render_displayResidual) {
      d_residual.Activate();
      glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
      texResidual.RenderToViewportFlipY();
    }

    // update parameters
    this->settings_pointCloudMode = settings_pointCloudMode.Get();

    this->settings_showActiveConstraints = settings_showActiveConstraints.Get();
    this->settings_showAllConstraints = settings_showAllConstraints.Get();
    this->settings_showCurrentCamera = settings_showCurrentCamera.Get();
    this->settings_showKFCameras = settings_showKFCameras.Get();
    this->settings_showTrajectory = settings_showTrajectory.Get();
    this->settings_showFullTrajectory = settings_showFullTrajectory.Get();

    setting_render_display3D = settings_show3D.Get();
    setting_render_displayDepth = settings_showLiveDepth.Get();
    setting_render_displayVideo = settings_showLiveVideo.Get();
#ifndef USE_MULTI_CAM
    setting_render_displayResidual = settings_showLiveResidual.Get();
#else
    setting_render_displayResidual = false;
#endif
    setting_render_renderWindowFrames = settings_showFramesWindow.Get();
    setting_render_plotTrackingFull = settings_showFullTracking.Get();
    setting_render_displayCoarseTrackingFull = settings_showCoarseTracking.Get();

    this->settings_absVarTH = settings_absVarTH.Get();
    this->settings_scaledVarTH = settings_scaledVarTH.Get();
    this->settings_minRelBS = settings_minRelBS.Get();
    this->settings_sparsity = settings_sparsity.Get();

    setting_desiredPointDensity = settings_nPts.Get();
    setting_desiredImmatureDensity = settings_nCandidates.Get();
    setting_maxFrames = settings_nMaxFrames.Get();
    setting_kfGlobalWeight = settings_kfFrequency.Get();
    setting_minGradHistAdd = settings_gradHistAdd.Get();

    if (settingsUtil) {
      settingsUtil->updatePangolinSettings();
    }

    if (settings_resetButton.Get()) {
      printf("RESET!\n");
      settings_resetButton.Reset();
      setting_fullResetRequested = true;
    }

    // Swap frames and Process Events
    pangolin::FinishFrame();

    if (needReset) reset_internal();

    if (pangolin::ShouldQuit()) {
      shouldQuitVar = true;
    }
  }

  printf("QUIT Pangolin thread!\n");
  printf("So Long, and Thanks for All the Fish!\n");
}

void PangolinDSOViewer::close() { running = false; }

void PangolinDSOViewer::join() {
  close();
  if (runThread.joinable()) runThread.join();
  printf("JOINED Pangolin thread!\n");
}

void PangolinDSOViewer::reset() { needReset = true; }

void PangolinDSOViewer::reset_internal() {
  model3DMutex.lock();
  for (size_t i = 0; i < keyframes.size(); i++) delete keyframes[i];
  keyframes.clear();
  allFramePoses.clear();
  keyframesByKFID.clear();
  connections.clear();
  model3DMutex.unlock();

  openImagesMutex.lock();
  internalVideoImg->setBlack();
  internalKFImg->setBlack();
  internalResImg->setBlack();
  videoImgChanged = kfImgChanged = resImgChanged = true;
  openImagesMutex.unlock();

  needReset = false;
}

void PangolinDSOViewer::drawConstraints() {
  if (settings_showAllConstraints) {
    // draw constraints
    glLineWidth(1);
    glBegin(GL_LINES);

    glColor3f(0, 1, 0);
    glBegin(GL_LINES);
    for (unsigned int i = 0; i < connections.size(); i++) {
      if (connections[i].to == 0 || connections[i].from == 0) continue;
      int nAct = connections[i].bwdAct + connections[i].fwdAct;
      int nMarg = connections[i].bwdMarg + connections[i].fwdMarg;
      if (nAct == 0 && nMarg > 0) {
        Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
        glVertex3f((GLfloat)t[0], (GLfloat)t[1], (GLfloat)t[2]);
        t = connections[i].to->camToWorld.translation().cast<float>();
        glVertex3f((GLfloat)t[0], (GLfloat)t[1], (GLfloat)t[2]);
      }
    }
    glEnd();
  }

  if (settings_showActiveConstraints) {
    glLineWidth(3);
    glColor3f(0, 0, 1);
    glBegin(GL_LINES);
    for (unsigned int i = 0; i < connections.size(); i++) {
      if (connections[i].to == 0 || connections[i].from == 0) continue;
      int nAct = connections[i].bwdAct + connections[i].fwdAct;

      if (nAct > 0) {
        Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
        glVertex3f((GLfloat)t[0], (GLfloat)t[1], (GLfloat)t[2]);
        t = connections[i].to->camToWorld.translation().cast<float>();
        glVertex3f((GLfloat)t[0], (GLfloat)t[1], (GLfloat)t[2]);
      }
    }
    glEnd();
  }

  if (settings_showTrajectory) {
    float colorGreen[3] = {0, 1, 0};
    glColor3f(colorGreen[0], colorGreen[1], colorGreen[2]);
    glLineWidth(3);

    glBegin(GL_LINE_STRIP);
    for (unsigned int i = 0; i < keyframes.size(); i++) {
      glVertex3f((float)keyframes[i]->camToWorld.translation()[0], (float)keyframes[i]->camToWorld.translation()[1],
                 (float)keyframes[i]->camToWorld.translation()[2]);
    }
    glEnd();
  }

  if (settings_showFullTrajectory) {
    float colorRed[3] = {1, 0, 0};
    glColor3f(colorRed[0], colorRed[1], colorRed[2]);
    glLineWidth(3);

    glBegin(GL_LINE_STRIP);
    for (unsigned int i = 0; i < allFramePoses.size(); i++) {
      glVertex3f((float)allFramePoses[i][0], (float)allFramePoses[i][1], (float)allFramePoses[i][2]);
    }
    glEnd();
  }
}

void PangolinDSOViewer::publishGraph(
    const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>,
                   Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>>& connectivity) {
  if (!setting_render_display3D) return;
  if (disableAllDisplay) return;

  model3DMutex.lock();
  connections.resize(connectivity.size());
  int runningID = 0;
  int totalActFwd = 0, totalActBwd = 0, totalMargFwd = 0, totalMargBwd = 0;
  for (std::pair<uint64_t, Eigen::Vector2i> p : connectivity) {
    int host = (int)(p.first >> 32);
    int target = (int)(p.first & (uint64_t)0xFFFFFFFF);

    assert(host >= 0 && target >= 0);
    if (host == target) {
      assert(p.second[0] == 0 && p.second[1] == 0);
      continue;
    }

    if (host > target) continue;

    connections[runningID].from = keyframesByKFID.count(host) == 0 ? 0 : keyframesByKFID[host];
    connections[runningID].to = keyframesByKFID.count(target) == 0 ? 0 : keyframesByKFID[target];
    connections[runningID].fwdAct = p.second[0];
    connections[runningID].fwdMarg = p.second[1];
    totalActFwd += p.second[0];
    totalMargFwd += p.second[1];

    uint64_t inverseKey = (((uint64_t)target) << 32) + ((uint64_t)host);
    if (connectivity.find(inverseKey) != connectivity.end() || true) {
      Eigen::Vector2i st = connectivity.at(inverseKey);
      connections[runningID].bwdAct = st[0];
      connections[runningID].bwdMarg = st[1];

      totalActBwd += st[0];
      totalMargBwd += st[1];

      runningID++;
    }
  }

  model3DMutex.unlock();
}

void PangolinDSOViewer::publishKeyframes(std::vector<FrameHessian*>& frames, bool final, CalibHessian* HCalib) {
  if (!setting_render_display3D) return;
  if (disableAllDisplay) return;

  boost::unique_lock<boost::mutex> lk(model3DMutex);
  for (FrameHessian* fh : frames) {
    if (keyframesByKFID.find(fh->frameID) == keyframesByKFID.end()) {
      KeyFrameDisplay* kfd = new KeyFrameDisplay(p_multi_camera);
      keyframesByKFID[fh->frameID] = kfd;
      keyframes.push_back(kfd);
    }
    keyframesByKFID[fh->frameID]->setFromKF(fh, HCalib);
  }
}

void PangolinDSOViewer::publishCamPose(FrameShell* frame, CalibHessian* HCalib) {
  if (!setting_render_display3D) return;
  if (disableAllDisplay) return;

  boost::unique_lock<boost::mutex> lk(model3DMutex);
  struct timeval time_now;
  gettimeofday(&time_now, NULL);
  lastNTrackingMs.push_back(
      ((time_now.tv_sec - last_track.tv_sec) * 1000.0f + (time_now.tv_usec - last_track.tv_usec) / 1000.0f));
  if (lastNTrackingMs.size() > 10) lastNTrackingMs.pop_front();
  last_track = time_now;

  if (!setting_render_display3D) return;

  this->HCalib = HCalib;

  currentCam->setFromF(frame, HCalib);
  allFramePoses.push_back(frame->camToWorld.translation().cast<float>());
}

void PangolinDSOViewer::pushLiveFrame(FrameHessian* image) {
  if (!setting_render_displayVideo) return;
  if (disableAllDisplay) return;

  boost::unique_lock<boost::mutex> lk(openImagesMutex);

  float alpha = 1.0f;

  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    float dt_len = image->max_dt_dx_dy[0][cid][0] - image->min_dt_dx_dy[0][cid][0];
    Vec3f* dt_dx_dy_start = image->dt_dx_dy[0] + wG[0] * hG[0] * cid;
    Vec2i* edge_label_image_start = image->edge_label_image[0] + wG[0] * hG[0] * cid;
    for (int i = 0; i < w * h; i++) {
      float gray_val = image->dI[i + w * h * cid][0] * 0.8 > 255.0f ? 255.0 : image->dI[i + w * h * cid][0] * 0.8;
#ifndef USE_EDGE_ALIGN
      internalVideoImg->data[i + w * h * cid][0] = internalVideoImg->data[i + w * h * cid][1] =
          internalVideoImg->data[i + w * h * cid][2] = gray_val;
#else
      float dt = (*(dt_dx_dy_start + i))[0];
      float dt_ratio = (dt_len > 0) ? (dt - image->min_dt_dx_dy[0][cid][0]) / dt_len : 0;
      dt = (dt_len > 0) ? 255.0f * (dt - image->min_dt_dx_dy[0][cid][0]) / dt_len : 0;
      float edge_val = (float)(*(edge_label_image_start + i))[0];
      bool is_edge_pixel = edge_val > 200.f;

      if (dt < 0) dt = 0;
      if (dt > 255) dt = 255;
      if (edge_val < 0) edge_val = 0;
      if (edge_val > 255) edge_val = 255;
      internalVideoImg->data[i + w * h * cid] =
          Vec3b(is_edge_pixel ? 0 : (1.0f - dt_ratio) * alpha * gray_val, is_edge_pixel ? 0 : alpha * gray_val,
                is_edge_pixel ? edge_val : (1.0f - dt_ratio) * alpha * gray_val);

#endif
    }
    internalVideoImg->putText(20, 20, std::to_string(int(image->mean_gray_val_each[cid])).c_str(), Vec3b(0, 255, 255),
                              cid);
  }
  videoImgChanged = true;
}

bool PangolinDSOViewer::needPushDepthImage() { return setting_render_displayDepth; }

void PangolinDSOViewer::pushDepthImage(MinimalImageB3* image, std::array<float, kCameraNumUsed> mean_gray_val) {
  if (!setting_render_displayDepth) return;
  if (disableAllDisplay) return;

  boost::unique_lock<boost::mutex> lk(openImagesMutex);

  struct timeval time_now;
  gettimeofday(&time_now, NULL);
  lastNMappingMs.push_back(
      ((time_now.tv_sec - last_map.tv_sec) * 1000.0f + (time_now.tv_usec - last_map.tv_usec) / 1000.0f));
  if (lastNMappingMs.size() > 10) lastNMappingMs.pop_front();
  last_map = time_now;

  memcpy(internalKFImg->data, image->data, w * h * 3 * kCameraNumUsed);
  // for (int cid = 0; cid < kCameraNumUsed; ++cid) {
  //   internalKFImg->putText(20, 20,
  //   std::to_string(int(mena_gray_val[cid])).c_str(), Vec3b(0, 255,255),cid);
  // }
  kfImgChanged = true;
}

void PangolinDSOViewer::publishTransformDSOToIMU(const dmvio::TransformDSOToIMU& transformDSOToIMUPassed) {
  if (!setting_render_display3D) return;
  if (disableAllDisplay) return;

  boost::unique_lock<boost::mutex> lk(model3DMutex);
  transformDSOToIMU =
      std::make_unique<dmvio::TransformDSOToIMU>(transformDSOToIMUPassed, std::make_shared<bool>(false),
                                                 std::make_shared<bool>(false), std::make_shared<bool>(false));
}

void PangolinDSOViewer::publishSystemStatus(dmvio::SystemStatus systemStatus) {
  boost::unique_lock<boost::mutex> lk(model3DMutex);
  this->systemStatus = systemStatus;
}

void PangolinDSOViewer::addGTCamPose(const Sophus::SE3& gtPose) {
  boost::unique_lock<boost::mutex> lk(model3DMutex);

  if (!setting_render_display3D || !HCalib) return;

  std::cout << "GTPose: " << gtPose.translation().transpose() << std::endl;

  if (!gtCamPoseSet) {
    firstGTCamPoseMetric = gtPose;
    firstCamPoseDSO = currentCam->camToWorld;  // Needed to make sure the first
                                               // pose is the same for both.
  }

  gtCamPoseMetric = gtPose;
  gtCamPoseSet = true;
  updateDisplayedCamPose();
}

// Caller should aquire model lock for us.
void PangolinDSOViewer::updateDisplayedCamPose() {
  if (!gtCamPoseSet || !transformDSOToIMU) return;
  if (!setting_render_display3D || !HCalib) return;

  // The visualizer shows cam to world in dso scale. The groundtruth pose is imu
  // to world in metric scale. This transforms to worldToCam
  SE3 worldToCam(transformDSOToIMU->transformPoseInverse(gtCamPoseMetric.matrix()));

  SE3 firstGTWorldToCam(transformDSOToIMU->transformPoseInverse(firstGTCamPoseMetric.matrix()));

  // We want the first pose to stay the same:
  // firstPose = offset * gtFirstPose;
  // --> offset = firstPose * gtFirstPose^(-1)
  SE3 offset = firstCamPoseDSO * firstGTWorldToCam;
  SE3 gtPoseTransformed = offset * worldToCam.inverse();

  currentGTCam->setFromPose(gtPoseTransformed, HCalib);
}

bool PangolinDSOViewer::shouldQuit() { return shouldQuitVar; }

}  // namespace IOWrap
}  // namespace dso
