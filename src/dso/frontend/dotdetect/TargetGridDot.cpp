//#define DEBUGDOT
//#define DEBUGDOTLINE

#include "TargetGridDot.h"

#include <algorithm>
#include <deque>
#include <map>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <utility>

#include "DetectPlateInfo.h"
#include "RandomGrid.h"
namespace dso::DotDetect {

bool TargetGridDot::FindBoardsKd(std::vector<Vertex>& conics, std::vector<std::set<Vertex*>>& multiPlate,
                                 const ParamsImageProcessing& params, const cv::Mat* img) {
  std::vector<Vertex*> left;
  for (Vertex& cur : conics) {
    left.emplace_back(&cur);
  }

  DetectPlateInfo curPlate(left, params, img);
  curPlate.FindBoards(multiPlate);
  curPlate.ResetVer();
  return true;
}

bool TargetGridDot::FindTarget(std::vector<Conic>& conics, std::map<int, std::vector<Conic*>>& plateConics,
                               const TargetGridInfo& boardInfo, const ParamsImageProcessing& params,
                               const cv::Mat* img) {
  std::vector<Vertex> vertexs, unused;
  for (int idx = 0; idx < conics.size(); ++idx) {
    vertexs.emplace_back(idx, &conics[idx]);
  }
  std::vector<std::set<Vertex*>> multiPlate;

  FindBoardsKd(vertexs, multiPlate, params, img);
#ifdef DEBUGDOT
  std::cerr << "multiPlate num: " << multiPlate.size() << std::endl;
  for (std::set<Vertex*>& onePlate : multiPlate) {
    std::cerr << "plate size:" << onePlate.size() << std::endl;
    cv::Mat plateShows;
    cv::cvtColor(*img, plateShows, cv::COLOR_GRAY2BGR);
    for (Vertex* oneVer : onePlate) {
      cv::circle(plateShows, cv::Point(oneVer->pc.x(), oneVer->pc.y()), 3, cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("plate pic", plateShows);
    cv::waitKey(0);
  }
#endif

  for (std::set<Vertex*>& onePlate : multiPlate) {
    FindTarget(onePlate, plateConics, boardInfo, params, img);
  }

  return true;
}

bool TargetGridDot::FindTarget(std::set<Vertex*>& plateVertex, std::map<int, std::vector<Conic*>>& plateConics,
                               TargetGridInfo boardInfo, const ParamsImageProcessing& params, const cv::Mat* img) {
  DetectPlateInfo curPlate(plateVertex, params, img);
  if (!curPlate.GetVerNeighborsAndCenter()) return false;
  if (!curPlate.GetMapGridEllipse()) return false;

  int plateId = -1, Plate_PG_idx;
  // Correlation of what we have with binary pattern
  const bool found = boardInfo.Match(curPlate.map_grid_ellipse_, plateId, Plate_PG_idx);

  if (!found) {
    //    std::cerr << "Pattern not found" << std::endl;
    return false;
  }

  if (plateConics.count(plateId) == 0) {
    plateConics.emplace(plateId, std::vector<Conic*>());
  }

  cv::Mat idMat;
  if (img) {
    cv::cvtColor(*img, idMat, cv::COLOR_GRAY2BGR);
  }

  // assign conic value
  for (Vertex* v : curPlate.vs_) {
    if (0 <= v->pg(0) && v->pg(0) < boardInfo.grid_size_(0) && 0 <= v->pg(1) && v->pg(1) < boardInfo.grid_size_(1) &&
        v->value >= 0 && !v->triples.empty()) {  //&& !v.triples.empty()
      // filter by nearby points
      for (Triple& curTriple : v->triples) {
        if (curTriple.Neighbour(0).HasGridPosition()) {
          Eigen::Vector2i nearDis = v->pg - curTriple.Neighbour(0).pg;
          if (abs(nearDis.x()) > 1 && abs(nearDis.y()) > 1) {
            //            std::cerr << "wrong point: " << v->pc.transpose() <<
            //            std::endl;
            continue;
          }
        }
      }
      v->conic->label = boardInfo.Plate_PG_Label.at(plateId)[Plate_PG_idx](v->pg(1), v->pg(0));
      plateConics.at(plateId).emplace_back(v->conic);
      v->conic->pos.x() = v->pg(1) * boardInfo.grid_spacing_;
      v->conic->pos.y() = v->pg(0) * boardInfo.grid_spacing_;
      v->conic->pos.z() = 0;

      if (img) {
        cv::circle(idMat, cv::Point(v->pc.x(), v->pc.y()), 2, cv::Scalar(0, 255, 0), 2);
        cv::putText(idMat, std::to_string(v->conic->label), cv::Point(v->pc.x(), v->pc.y()), 1, 1,
                    cv::Scalar(0, 0, 255));
      }
    }
  }
  if (img) {
    cv::imshow("point id", idMat);
    cv::waitKey(0);
  }
  return true;
}

}  // namespace dso::DotDetect