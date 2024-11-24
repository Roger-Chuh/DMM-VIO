#pragma once

#include <Eigen/Eigen>
#include <algorithm>
#include <array>
#include <list>
#include <set>
#include <vector>

#include "Conic.h"
namespace dso {

namespace DotDetect {

const static int GRID_INVALID = std::numeric_limits<int>::min();

struct Triple;

struct Vertex {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  inline Vertex(size_t id, Conic *const c)
      : id(id), conic(c), pc(c->center), pg(GRID_INVALID, GRID_INVALID),
        area(0.0), value(-1) {}

  inline bool HasGridPosition() { return pg(0) != GRID_INVALID; }

  inline void ResetGridPosition() {
    pg(0) = GRID_INVALID;
    pg(1) = GRID_INVALID;
  }

  inline void ResetVertex() {
    used = false;
    ResetGridPosition();
    //    triples.clear();
    //    neighbours.clear();
  }

  size_t id;
  Conic *const conic;
  Eigen::Vector2d pc;
  Eigen::Vector2i pg;
  std::vector<Triple> triples;
  std::set<Vertex *> neighbours;
  double area;
  int value;

  bool used = false;

  double distance(const Vertex &other) const {
    Vec2 diff = pc - other.pc;
    return diff.norm();
  }
};

struct Triple {
  inline Triple(Vertex &o1, Vertex &c, Vertex &o2) { vs = {&o1, &c, &o2}; }

  Triple(const Triple &triple) = default;

  inline Vertex &Center() { return *vs[1]; }
  inline const Vertex &Center() const { return *vs[1]; }

  inline Vertex &Neighbour(size_t i) { return i ? *vs[2] : *vs[0]; }
  inline const Vertex &Neighbour(size_t i) const { return i ? *vs[2] : *vs[0]; }

  inline Vertex &OtherNeighbour(size_t i) { return i ? *vs[0] : *vs[2]; }
  inline const Vertex &OtherNeighbour(size_t i) const {
    return i ? *vs[0] : *vs[2];
  }

  inline Vertex &Vert(size_t i) { return *vs[i]; }
  inline const Vertex &Vert(size_t i) const { return *vs[i]; }

  inline Eigen::Vector2d Dir() const { return vs[2]->pc - vs[0]->pc; }

  inline bool Contains(const Vertex &v) const {
    const bool found = std::find(vs.begin(), vs.end(), &v) != vs.end();
    return found;
  }

  inline bool In(const std::set<Vertex *> bag) const {
    return bag.find(vs[0]) != bag.end() && bag.find(vs[2]) != bag.end();
  }

  inline void Reverse() { std::swap(vs[0], vs[2]); }

  // Colinear sequence of vertices, v[0], v[1], v[2]. v[1] is center
  std::vector<Vertex *> vs;
};

inline bool operator==(const Vertex &lhs, const Vertex &rhs) {
  return lhs.id == rhs.id;
}

inline bool AreCollinear(const Triple &t1, const Triple &t2) {
  return t1.Contains(t2.Center()) && t2.Contains(t1.Center());
}

inline double Distance(const Vertex &v1, const Vertex &v2) {
  return (v2.pc - v1.pc).norm();
}

inline std::ostream &operator<<(std::ostream &os, const Vertex &v) {
  os << "(" << v.pg.transpose() << ")";
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const Triple &t) {
  os << t.Vert(0) << " - " << t.Vert(1) << " - " << t.Vert(2);
  return os;
}

struct LineGroup {
  explicit LineGroup(const Triple &o)
      : ops({o.vs[0]->id, o.vs[1]->id, o.vs[2]->id}) {}

  bool Merge(LineGroup &o) {
    if (last() == o.second() && pen() == o.first()) {
      ops.insert(ops.end(), std::next(o.ops.begin(), 2), o.ops.end());
      o.ops.clear();
      return true;
    } else if (o.last() == second() && o.pen() == first()) {
      ops.insert(ops.begin(), o.ops.begin(), std::prev(o.ops.end(), 2));
      o.ops.clear();
      return true;
    } else if (last() == o.pen() && pen() == o.last()) {
      ops.insert(ops.end(), std::next(o.ops.rbegin(), 2), o.ops.rend());
      o.ops.clear();
      return true;
    } else if (first() == o.second() && second() == o.first()) {
      ops.insert(ops.begin(), o.ops.rbegin(), std::prev(o.ops.rend(), 2));
      o.ops.clear();
      return true;
    }
    return false;
  }

  void Reverse() {
    std::list<size_t> rev_ops;
    rev_ops.insert(rev_ops.begin(), ops.rbegin(), ops.rend());
    ops = rev_ops;
  }

  size_t first() { return *ops.begin(); }
  size_t last() { return *ops.rbegin(); }
  size_t second() { return *std::next(ops.begin()); }
  size_t pen() { return *std::next(ops.rbegin()); }

  std::list<size_t> ops;
};

struct KDNode {
  Vertex *point;
  std::unique_ptr<KDNode> left;
  std::unique_ptr<KDNode> right;

  explicit KDNode(Vertex *p) : point(p), left(nullptr), right(nullptr) {}
};

class KDTree {
public:
  KDTree() : root(nullptr) {}

  void insert(Vertex *point) {
    root = insertRecursive(std::move(root), point, 0);
  }

  KDNode *getRoot() const { return root.get(); }

  std::vector<KDNode *> getMiddleOrder() {
    std::vector<KDNode *> nodes;
    inOrderTraversal(root.get(), nodes);
    return nodes;
  }

  std::vector<KDNode *> getPreOrder() {
    std::vector<KDNode *> result;
    preOrder(root.get(), result);
    return result;
  }

  std::vector<KDNode *> getPostOrder() {
    std::vector<KDNode *> result;
    postOrder(root.get(), result);
    return result;
  }

  std::vector<Vertex *> searchNearestNeighbors(const Vertex &query, size_t k) {
    std::vector<Vertex *> nearestNeighbors;
    searchRecursive(root.get(), query, k, nearestNeighbors, 0);
    return nearestNeighbors;
  }

private:
  std::unique_ptr<KDNode> root;

  //中序遍历
  void inOrderTraversal(KDNode *node, std::vector<KDNode *> &nodes) {
    if (!node) {
      return;
    }
    inOrderTraversal(node->left.get(), nodes);
    nodes.push_back(node);
    inOrderTraversal(node->right.get(), nodes);
  }

  void preOrder(KDNode *node, std::vector<KDNode *> &result) {
    if (!node) {
      return;
    }

    result.push_back(node);
    preOrder(node->left.get(), result);
    preOrder(node->right.get(), result);
  }

  void postOrder(KDNode *node, std::vector<KDNode *> &result) {
    if (!node) {
      return;
    }

    postOrder(node->left.get(), result);
    postOrder(node->right.get(), result);
    result.push_back(node);
  }

  std::unique_ptr<KDNode> insertRecursive(std::unique_ptr<KDNode> node,
                                          Vertex *point, size_t depth) {
    if (!node) {
      return std::make_unique<KDNode>(point);
    }

    size_t axis = depth % 2;

    if (point->pc(axis) < node->point->pc(axis)) {
      node->left = insertRecursive(std::move(node->left), point, depth + 1);
    } else {
      node->right = insertRecursive(std::move(node->right), point, depth + 1);
    }

    return node;
  }

  void searchRecursive(const KDNode *node, const Vertex &query, size_t k,
                       std::vector<Vertex *> &nearestNeighbors,
                       size_t depth) const {
    if (!node) {
      return;
    }

    size_t axis = depth % 2;

    // 根据 query 的位置，决定是先访问左子树还是右子树
    const KDNode *firstChild = nullptr;
    const KDNode *secondChild = nullptr;
    if (query.pc(axis) < node->point->pc(axis)) {
      firstChild = node->left.get();
      secondChild = node->right.get();
    } else {
      firstChild = node->right.get();
      secondChild = node->left.get();
    }

    searchRecursive(firstChild, query, k, nearestNeighbors, depth + 1);

    // 检查当前节点是否更近
    if (nearestNeighbors.size() < k ||
        query.distance(*node->point) <
            query.distance(*nearestNeighbors.back())) {
      nearestNeighbors.push_back(node->point);
      std::sort(nearestNeighbors.begin(), nearestNeighbors.end(),
                [&query](const Vertex *a, const Vertex *b) {
                  return query.distance(*a) < query.distance(*b);
                });
      if (nearestNeighbors.size() > k) {
        nearestNeighbors.pop_back();
      }
    }

    // 如果当前节点到 query 的距离小于最远邻居的距离，继续搜索另一个子树
    if (nearestNeighbors.size() < k ||
        std::abs(query.pc(axis) - node->point->pc(axis)) <
            query.distance(*nearestNeighbors.back())) {
      searchRecursive(secondChild, query, k, nearestNeighbors, depth + 1);
    }
  }
};

} // namespace DotDetect
} // namespace dso