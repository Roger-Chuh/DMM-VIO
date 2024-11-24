#ifndef YVR_CALIB_DETECTPLATEINFO_H
#define YVR_CALIB_DETECTPLATEINFO_H
#include "BasicStruct.h"
#include "ImagesProcessParams.h"
#include "LineGroup.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

//#define DEBUGDOT
//#define DEBUGDOTLINE

namespace dso::DotDetect {
class DetectPlateInfo {
public:
  std::vector<Vertex *> vs_;
  const ParamsImageProcessing &params_;
  const cv::Mat *img_;
  std::map<Eigen::Vector2i const, Vertex *, lessCompare,
           Eigen::aligned_allocator<std::pair<Eigen::Vector2i const, Vertex *>>>
      map_grid_ellipse_;

private:
  Vertex *central_ = nullptr;
  std::vector<Triple *> principle_;

public:
  DetectPlateInfo(std::vector<Vertex *> &vs,
                  const ParamsImageProcessing &params, const cv::Mat *img)
      : vs_(vs), params_(params), img_(img) {}
  DetectPlateInfo(std::set<Vertex *> &vs, const ParamsImageProcessing &params,
                  const cv::Mat *img)
      : params_(params), img_(img) {
    for (Vertex *oneVer : vs) {
      vs_.emplace_back(oneVer);
    }
  }

  void ResetVer() {
    for (Vertex *oneVer : vs_) {
      oneVer->ResetVertex();
    }
  }

  bool GetVerNeighborsAndCenter() {
    if (vs_.empty())
      return false;
    Conic temp;
    temp.center.setZero();
    Vertex center(-1, &temp);
    KDTree tree;
    for (Vertex *v : vs_) {
      tree.insert(v);
      center.pc += v->pc;
    }
    center.pc /= (int)vs_.size();

    //    size_t k = 9;  // include self
    //    for (Vertex* ver : vs_) {
    //      std::vector<Vertex*> nearestNeighbors =
    //      tree.searchNearestNeighbors(*ver, k);
    //      GetNeighborsTriples(nearestNeighbors, params_.max_line_dist_ratio,
    //      params_.max_norm_triple_area, img_);
    //    }

    std::vector<Vertex *> nearestNeighbors =
        tree.searchNearestNeighbors(center, std::max(100, (int)vs_.size()));

    for (Vertex *v : nearestNeighbors) {
      if (v->triples.size() >= 2) {
        principle_ = PrincipleDirections(*v);
        if (principle_.size() == 2) {
          central_ = v;
          v->used = true;
          break;
        }
      }
    }

    if (!central_) {
      return false;
    }

#ifdef DEBUGDOT
    std::cerr << "No central point found. nearestNeighbors: "
              << nearestNeighbors.size() << std::endl;
    cv::Mat tripleMat;
    cv::cvtColor(*img_, tripleMat, cv::COLOR_GRAY2BGR);
    cv::rectangle(tripleMat,
                  cv::Point(central_->conic->bbox.x1, central_->conic->bbox.y1),
                  cv::Point(central_->conic->bbox.x2, central_->conic->bbox.y2),
                  cv::Scalar(255, 0, 0));
    cv::imshow("one board triple with center", tripleMat);
    cv::waitKey(0);
#endif

    return true;
  }

  bool GetMapGridEllipse() {
    if (central_ == nullptr || principle_.empty())
      return false;
    // Search structures
    std::deque<Vertex *> fringe;
    std::deque<Vertex *> available;
    for (Vertex *v : vs_) {
      available.push_back(v);
    }
    // Setup central as center of grid
    SetGrid(*central_, Eigen::Vector2i(0, 0));
    available.erase(std::find(available.begin(), available.end(), central_));
    // add neighbours of central to form basis
    for (int idx = 0; idx < 2; ++idx) {
      Triple &t = *principle_[idx];
      Eigen::Vector2i g(0, 0);

      for (int j = 0; j < 2; ++j) {
        Vertex &n = t.Neighbour(j);
        g[idx] = 2 * j - 1;
        SetGrid(n, g);
        available.erase(std::find(available.begin(), available.end(), &n));
        fringe.push_back(&n);
      }
    }

    // depth first search extending 'fringe' set by adding colinear vertices
    while (!fringe.empty()) {
      Vertex &f = *fringe.front();
      for (size_t idx = 0; idx < f.triples.size(); ++idx) {
        Triple &t = f.triples[idx];
        for (size_t j = 0; j < 2; ++j) {
          Vertex &n = t.Neighbour(j);
          Vertex &no = t.OtherNeighbour(j);
          if (n.HasGridPosition()) {
            // expected other-neighbour grid position
            const Eigen::Vector2i step = f.pg - n.pg;
            const Eigen::Vector2i go = f.pg + step;

            // Only accept local neighbours.
            if (std::abs(step[0]) > 1 || std::abs(step[1]) > 1) {
              continue;
            }

            // Either check consistent or complete
            if (no.HasGridPosition()) {
              // check
              if (no.pg != go) {
                // tracking bad!
                no.ResetGridPosition();
                DelGrid(no.pg);
                continue;
              }
            } else {
              // add
              SetGrid(no, go);
              fringe.push_back(&no);
            }
            // no need to check other neighbour
            break;
          }
        }
      }
      // Remove from fringe
      fringe.pop_front();
    }

    // Try to add any that we've missed by 'filling in'
    while (!available.empty()) {
      Vertex &f = *available.front();
      for (size_t idx = 0; idx < f.triples.size(); ++idx) {
        Triple &t = f.triples[idx];
        Vertex &n = t.Neighbour(0);
        Vertex &no = t.OtherNeighbour(0);

        if (n.HasGridPosition() && no.HasGridPosition()) {
          const Eigen::Vector2i step = no.pg - n.pg;
          if (step[0] % 2 == 0 && step[1] % 2 == 0) {
            const Eigen::Vector2i g = (no.pg + n.pg) / 2;
            if (f.HasGridPosition()) {
              // check
              if (f.pg != g) {
                // tracking bad.
                f.ResetGridPosition();
                DelGrid(f.pg);
                continue;
              }
            } else {
              SetGrid(f, g);
              continue;
            }
          }
        }
      }
      available.pop_front();
    }

#ifdef DEBUGDOT
    cv::Mat boxMat;
    cv::cvtColor(*img_, boxMat, cv::COLOR_GRAY2BGR);
    for (const auto &label : map_grid_ellipse_) {
      cv::rectangle(boxMat,
                    cv::Point2i(label.second->conic->bbox.x1,
                                label.second->conic->bbox.y1),
                    cv::Point2i(label.second->conic->bbox.x2,
                                label.second->conic->bbox.y2),
                    cv::Scalar(0, 0, 255));
    }
    cv::imshow("grid box in use: map grid & center point", boxMat);
    cv::waitKey(0);
#endif

    // Compute area and grid neighbours for all ellipses in grid
    for (auto &map : map_grid_ellipse_) {
      Vertex &v = *map.second;
      v.area = Area(*v.conic);
      v.neighbours = Neighbours(map_grid_ellipse_, v);
    }

    // Determine binary value from neighbours area
    for (auto &grip : map_grid_ellipse_) {
      Vertex &v = *grip.second;

      if (v.neighbours.size() > 2) {
        // just take min/max - no need to sort
        // Sort neightbours by circle area
        std::vector<Dist> vecrad;
        vecrad.push_back(Dist{&v, v.area});
        for (Vertex *n : v.neighbours) {
          vecrad.push_back(Dist{n, n->area});
        }
        std::sort(vecrad.begin(), vecrad.end());

        const double _area0 = vecrad.front().dist;
        const double _area1 = vecrad.back().dist;

        // determine these values from pattern
        const double area0 = 2 * 2;
        const double area1 = 3 * 3;
        if (area1 * _area0 / area0 <= _area1 * 1.2) {
          // is difference
          const double d0 = std::abs(v.area - _area0);
          const double d1 = std::abs(v.area - _area1);

          if (std::abs((d0 - d1) / (d0 + d1)) > 0.25) {
            v.value = (d0 < d1) ? 0 : 1;
          } else {
            v.value = -1;
          }
        }
      } else {
        v.value = -1;
      }
    }

    return true;
  }

  bool FindBoards(std::vector<std::set<Vertex *>> &multiPlate) {
    KDTree tree;
    for (Vertex *v : vs_) {
      tree.insert(v);
    }

    std::unordered_map<Vertex *, int> vertConnectCount;
    size_t k = 9; // include self
    for (Vertex *ver : vs_) {
      std::vector<Vertex *> nearestNeighbors =
          tree.searchNearestNeighbors(*ver, k);
      GetNeighborsTriples(nearestNeighbors, params_.max_line_dist_ratio,
                          params_.max_norm_triple_area, img_);
      for (Triple &oneTri : ver->triples) {
        for (int i = 0; i < 2; ++i) {
          if (vertConnectCount.count(&oneTri.Neighbour(i)) == 0) {
            vertConnectCount.emplace(&oneTri.Neighbour(i), 1);
          } else {
            vertConnectCount.at(&oneTri.Neighbour(i)) += 1;
          }
        }
      }
    }

#ifdef DEBUGDOT
    cv::Mat lineImg;
    cv::cvtColor(img_->clone(), lineImg, cv::COLOR_GRAY2BGR);
    for (Vertex *oneVer : vs_) {
      for (Triple &oneTri : oneVer->triples)
        cv::line(lineImg,
                 cv::Point((int)oneTri.Neighbour(0).pc.x(),
                           (int)oneTri.Neighbour(0).pc.y()),
                 cv::Point((int)oneTri.Neighbour(1).pc.x(),
                           (int)oneTri.Neighbour(1).pc.y()),
                 cv::Scalar(0, 0, 255), 1);
      if (vertConnectCount.count(oneVer) > 0 &&
          vertConnectCount.at(oneVer) > 1) {
        cv::circle(lineImg, cv::Point((int)oneVer->pc.x(), (int)oneVer->pc.y()),
                   2, cv::Scalar(0, 255, 0), 2);
      }
    }
    cv::imshow("line mat", lineImg);
    cv::waitKey(0);
#endif

    for (std::pair<Vertex *const, int> oneVertCount : vertConnectCount) {
      if (oneVertCount.second <= 1 || oneVertCount.first->used ||
          oneVertCount.first->triples.size() < 2)
        continue;

      std::queue<Vertex *> queueDfs;
      queueDfs.emplace(oneVertCount.first);

#ifdef DEBUGDOT
      cv::Mat plateImg;
      cv::cvtColor(img_->clone(), lineImg, cv::COLOR_GRAY2BGR);
#endif

      std::set<Vertex *> onePlate;
      while (!queueDfs.empty()) {
        Vertex *curVer = queueDfs.front();
        onePlate.insert(curVer);
        queueDfs.pop();
        for (Triple &triple : curVer->triples) {
          for (int i = 0; i < 2; ++i) {
#ifdef DEBUGDOT
            cv::line(lineImg,
                     cv::Point((int)triple.Neighbour(0).pc.x(),
                               (int)triple.Neighbour(0).pc.y()),
                     cv::Point((int)triple.Neighbour(1).pc.x(),
                               (int)triple.Neighbour(1).pc.y()),
                     cv::Scalar(0, 155, 0));
#endif
            if (vertConnectCount.count(&triple.Neighbour(i)) > 0 &&
                vertConnectCount.at(&triple.Neighbour(i)) > 1 &&
                !triple.Neighbour(i).used) {
              triple.Neighbour(i).used = true;
              queueDfs.push(&triple.Neighbour(i));
              onePlate.insert(&triple.Neighbour(i));
#ifdef DEBUGDOT
              cv::circle(lineImg,
                         cv::Point((int)curVer->pc.x(), (int)curVer->pc.y()), 2,
                         cv::Scalar(0, 0, 255));
//              cv::imshow("plate img", lineImg);
//              cv::waitKey(0);
#endif
            }
          }
        }
      }
      if (onePlate.size() < 15)
        continue;
      multiPlate.emplace_back(onePlate);

#ifdef DEBUGDOT
      for (Vertex *oneVer : onePlate) {
        cv::circle(lineImg, cv::Point((int)oneVer->pc.x(), (int)oneVer->pc.y()),
                   3, cv::Scalar(0, 0, 255), 2);
      }
      cv::imshow("plate img", lineImg);
      cv::waitKey(0);
#endif
    }

    return true;
  }

private:
  void SetGrid(Vertex &v, const Eigen::Vector2i &g) {
    v.pg = g;
    map_grid_ellipse_[g] = &v;
    v.used = true;
  }
  void DelGrid(const Eigen::Vector2i &g) { map_grid_ellipse_.erase(g); }

  static double Area(const Conic &c) {
    // http://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
    const Eigen::Matrix2d A33 = c.C.block<2, 2>(0, 0);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(A33);
    if (eigensolver.info() == Eigen::Success) {
      const double detA = c.C.determinant();
      const double detA33 = A33.determinant();
      const double fac = -detA / (detA33);
      const Eigen::Vector2d &l = eigensolver.eigenvalues();

      const double a = sqrt(fac / l[0]);
      const double b = sqrt(fac / l[1]);

      return M_PI * a * b;
    } else {
      return 0.0;
    }
  }

  static std::vector<Triple *> PrincipleDirections(Vertex &v) {
    // Find principle directions by observing that neighbours from princple
    // directions are central within triple that is also formed from these
    // neighbours.
    std::set<Triple *> pd;
    for (size_t i = 0; i < v.triples.size(); ++i) {
      Triple &t = v.triples[i];
      for (size_t j = 0; j < 2; ++j) {
        Vertex &n = t.Neighbour(j);
        for (Triple &a : n.triples) {
          if (a.In(v.neighbours)) {
            // a is parallel to principle direction
            // t is a parallel direction.
            pd.insert(&t);
            break;
          }
        }
      }
    }

    // convert to vector
    std::vector<Triple *> ret;
    ret.insert(ret.begin(), pd.begin(), pd.end());

    // find most x-ily and y-ily
    if (ret.size() == 2) {
      Eigen::Vector2d d[2] = {ret[0]->Dir(), ret[1]->Dir()};
      if (std::abs(d[1][0]) > std::abs(d[0][0])) {
        std::swap(ret[0], ret[1]);
        std::swap(d[0], d[1]);
      }

      // place in axis ascending order.
      if (d[0][0] < 0)
        ret[0]->Reverse();
      if (d[1][1] < 0)
        ret[1]->Reverse();
    }

    return ret;
  }
  static bool GetNeighborsTriples(std::vector<Vertex *> &nearestNeighbors,
                                  double thresh_dist, double thresh_area,
                                  const cv::Mat *img) {
    if (nearestNeighbors.size() < 3)
      return false;
    // check neighbor exactly
    std::vector<bool> used;
    used.resize(nearestNeighbors.size(), false); // pretend overload
    double minDist =
        (nearestNeighbors[0]->pc - nearestNeighbors[1]->pc).norm() * 2;

#ifdef DEBUGDOTLINE
    cv::Mat lineMat;
    cv::cvtColor(*img, lineMat, cv::COLOR_GRAY2BGR);
    cv::rectangle(lineMat,
                  cv::Point2i(nearestNeighbors[0]->conic->bbox.x1,
                              nearestNeighbors[0]->conic->bbox.y1),
                  cv::Point2i(nearestNeighbors[0]->conic->bbox.x2,
                              nearestNeighbors[0]->conic->bbox.y2),
                  cv::Scalar(0, 255, 0));
    for (int i = 1; i < nearestNeighbors.size(); ++i) {
      cv::rectangle(lineMat,
                    cv::Point2i(nearestNeighbors[i]->conic->bbox.x1,
                                nearestNeighbors[i]->conic->bbox.y1),
                    cv::Point2i(nearestNeighbors[i]->conic->bbox.x2,
                                nearestNeighbors[i]->conic->bbox.y2),
                    cv::Scalar(255, 0, 0));
    }
#endif

    for (int n1 = 1; n1 < nearestNeighbors.size(); ++n1) {
      const double &d1 =
          (nearestNeighbors[n1]->pc - nearestNeighbors[0]->pc).norm();
      for (int n2 = n1 + 1; n2 < nearestNeighbors.size(); ++n2) {
        if (used[n1] || used[n2])
          continue;
        const double &d2 =
            (nearestNeighbors[n2]->pc - nearestNeighbors[0]->pc).norm();
        if (d2 > minDist) {
          used[n2] = true;
          continue;
        }
        if (!used[n1] && !used[n2] &&
            NormArea(nearestNeighbors[n1]->pc, nearestNeighbors[0]->pc,
                     nearestNeighbors[n2]->pc) <
                thresh_area) { // Check points are colinear with center
          if (2.0 * fabs(d2 - d1) / (fabs(d1) + fabs(d2)) <
              thresh_dist) { // Check distances are similar
            used[n1] = true;
            used[n2] = true;
            nearestNeighbors[0]->neighbours.insert(nearestNeighbors[n1]);
            nearestNeighbors[0]->neighbours.insert(nearestNeighbors[n2]);
            nearestNeighbors[0]->triples.emplace_back(*nearestNeighbors[n1],
                                                      *nearestNeighbors[0],
                                                      *nearestNeighbors[n2]);
#ifdef DEBUGDOTLINE
            cv::line(lineMat,
                     cv::Point2i(nearestNeighbors[0]->pc.x(),
                                 nearestNeighbors[0]->pc.y()),
                     cv::Point2i(nearestNeighbors[n1]->pc.x(),
                                 nearestNeighbors[n1]->pc.y()),
                     cv::Scalar(0, 0, 255));
            cv::line(lineMat,
                     cv::Point2i(nearestNeighbors[0]->pc.x(),
                                 nearestNeighbors[0]->pc.y()),
                     cv::Point2i(nearestNeighbors[n2]->pc.x(),
                                 nearestNeighbors[n2]->pc.y()),
                     cv::Scalar(0, 0, 255));
#endif
          }
        }
      }
    }

#ifdef DEBUGDOTLINE
    cv::imshow("lineMat:", lineMat);
    cv::waitKey(0);
#endif

    return true;
  }

  static double NormArea(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2,
                         const Eigen::Vector2d &p3) {
    // Compute signed area
    const double area = SignedArea(p1, p2, p3);
    const double len = (p3 - p1).norm();
    return std::abs(area) / (len * len);
  }

  static double SignedArea(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2,
                           const Eigen::Vector2d &p3) {
    return p1(0) * (p2(1) - p3(1)) + p2(0) * (p3(1) - p1(1)) +
           p3(0) * (p1(1) - p2(1));
  }

  static std::set<DotDetect::Vertex *> Neighbours(
      std::map<Eigen::Vector2i const, DotDetect::Vertex *, lessCompare,
               Eigen::aligned_allocator<
                   std::pair<Eigen::Vector2i const, DotDetect::Vertex *>>> &map,
      const Vertex &v) {
    std::set<DotDetect::Vertex *> neighbours;
    for (int r = -1; r <= 1; ++r) {
      for (int c = -1; c <= 1; ++c) {
        Eigen::Vector2i pg(v.pg[0] + c, v.pg[1] + r);
        auto i = map.find(pg);
        if (i != map.end()) {
          neighbours.insert(i->second);
        }
      }
    }
    return neighbours;
  }
};

} // namespace dso::DotDetect
#endif // YVR_CALIB_DETECTPLATEINFO_H
