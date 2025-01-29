#pragma once

#include <cassert>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace dso {
template <typename T, typename Alloc = std::allocator<T>>
class BufferContainer {
  template <bool B, class _T = void>
  using enable_if_t = typename std::enable_if<B, _T>::type;

public:
  using Type = T;
  using Ptr = std::shared_ptr<T>;
  using CallBackFunc = std::function<void(T &)>;

  BufferContainer() {
    new_alloc_size_ = 0;
    max_new_alloc_size_ = -1;
    malloc_callback_ = nullptr;
    acquire_callback_ = nullptr;
    recycle_callback_ = nullptr;
    is_init_ = false;
  }

  template <
      typename FuncM, typename FuncA, typename FuncR, typename... Args,
      typename = enable_if_t<std::is_convertible<FuncM, CallBackFunc>::value>,
      typename = enable_if_t<std::is_convertible<FuncA, CallBackFunc>::value>,
      typename = enable_if_t<std::is_convertible<FuncR, CallBackFunc>::value>,
      typename = enable_if_t<std::is_constructible<T, Args...>::value>>
  BufferContainer(FuncM malloc_callback, FuncA acquire_callback,
                  FuncR recycle_callback, int pre_alloc_size,
                  int max_new_alloc_size, Args &&... args)
      : new_alloc_size_(0), max_new_alloc_size_(-1),
        malloc_callback_(malloc_callback), acquire_callback_(acquire_callback),
        recycle_callback_(recycle_callback), is_init_(true) {
    Malloc(pre_alloc_size, max_new_alloc_size,
           std::forward<decltype(args)>(args)...);
  }

  ~BufferContainer() { Release(); }

  BufferContainer(const BufferContainer<T> &oth) = delete;
  BufferContainer<T> &operator=(const BufferContainer<T> &) = delete;

  BufferContainer(BufferContainer<T> &&oth) {
    std::unique_lock<std::mutex> lock(mtx_);
    buffers_ = std::move(oth.buffers_);
    points_ = std::move(oth.points_);
    new_alloc_size_ = oth.new_alloc_size_;
    max_new_alloc_size_ = oth.max_new_alloc_size_;
    malloc_callback_ = oth.malloc_callback_;
    acquire_callback_ = oth.acquire_callback_;
    recycle_callback_ = oth.recycle_callback_;
    is_init_ = oth.is_init_;
  }

  BufferContainer<T> &operator=(BufferContainer<T> &&oth) {
    std::unique_lock<std::mutex> lock(mtx_);
    buffers_ = std::move(oth.buffers_);
    points_ = std::move(oth.points_);
    new_alloc_size_ = oth.new_alloc_size_;
    max_new_alloc_size_ = oth.max_new_alloc_size_;
    malloc_callback_ = oth.malloc_callback_;
    acquire_callback_ = oth.acquire_callback_;
    recycle_callback_ = oth.recycle_callback_;
    is_init_ = oth.is_init_;
    return *this;
  }

  template <
      typename FuncM, typename FuncA, typename FuncR, typename... Args,
      typename = enable_if_t<std::is_convertible<FuncM, CallBackFunc>::value>,
      typename = enable_if_t<std::is_convertible<FuncA, CallBackFunc>::value>,
      typename = enable_if_t<std::is_convertible<FuncR, CallBackFunc>::value>,
      typename = enable_if_t<std::is_constructible<T, Args...>::value>>
  void Initial(FuncM malloc_callback, FuncA acquire_callback,
               FuncR recycle_callback, int pre_alloc_size,
               int max_new_alloc_size, Args &&... args) {
    if (IsInitial()) {
      return;
    }
    BindAllCallBack(malloc_callback, acquire_callback, recycle_callback);
    Malloc(pre_alloc_size, max_new_alloc_size,
           std::forward<decltype(args)>(args)...);
    SetInitial(true);
  }

  template <
      typename FuncM, typename FuncA, typename FuncR,
      typename = enable_if_t<std::is_convertible<FuncM, CallBackFunc>::value>,
      typename = enable_if_t<std::is_convertible<FuncA, CallBackFunc>::value>,
      typename = enable_if_t<std::is_convertible<FuncR, CallBackFunc>::value>>
  inline void BindAllCallBack(FuncM malloc_callback, FuncA acquire_callback,
                              FuncR recycle_callback) {
    std::unique_lock<std::mutex> lock(mtx_);
    malloc_callback_ = std::move(malloc_callback);
    acquire_callback_ = std::move(acquire_callback);
    recycle_callback_ = std::move(recycle_callback);
  }

  template <typename... Args,
            typename = enable_if_t<std::is_constructible<T, Args...>::value>>
  inline int Malloc(int pre_alloc_size, int max_new_alloc_size,
                    Args &&... args) {
    if (Free() == 0) {
      return 0;
    }
    std::unique_lock<std::mutex> lock(mtx_);
    max_new_alloc_size_ = max_new_alloc_size;
    if (pre_alloc_size > 0) {
      buffers_.resize((uint32_t)pre_alloc_size,
                      T(std::forward<decltype(args)>(args)...));
      for (uint32_t i = 0; i < (uint32_t)pre_alloc_size; ++i) {
        if (malloc_callback_) {
          malloc_callback_(buffers_[i]);
        }
        points_.push_back(&buffers_[i]);
      }
    }

    return 1;
  }

  template <typename... Args,
            typename = enable_if_t<std::is_constructible<T, Args...>::value>>
  inline Ptr Acquire(Args &&... args) {
    std::unique_lock<std::mutex> lock(mtx_);

    T *point = nullptr;
    if (points_.empty()) {
      if (max_new_alloc_size_ < 0 ||
          new_alloc_size_ < (uint32_t)max_new_alloc_size_) {
        point = new T(std::forward<decltype(args)>(args)...);
        if (malloc_callback_) {
          malloc_callback_(*point);
        }
        new_alloc_size_++;
      }
    } else {
      point = points_.front();
      points_.pop_front();
    }

    if (point && acquire_callback_) {
      acquire_callback_(*point);
    }
    return Ptr(point, [&](T *elem) { this->Recycle(elem); });
  }

  inline int Release() {
    if (Free() == 0) {
      return 0;
    }
    std::unique_lock<std::mutex> lock(mtx_);
    max_new_alloc_size_ = -1;
    malloc_callback_ = nullptr;
    acquire_callback_ = nullptr;
    recycle_callback_ = nullptr;
    is_init_ = false;
    return 1;
  }

  inline uint32_t Size() {
    std::unique_lock<std::mutex> lock(mtx_);
    return points_.size();
  }

  inline bool IsNewAllocValid() {
    std::unique_lock<std::mutex> lock(mtx_);
    return max_new_alloc_size_ < 0 ||
           new_alloc_size_ < (uint32_t)max_new_alloc_size_;
  }

  inline void SetInitial(bool tag) { is_init_ = tag; }

  inline bool IsInitial() { return is_init_; }

private:
  inline void Recycle(T *elem) {
    if (!elem) {
      return;
    }

    if (recycle_callback_) {
      recycle_callback_(*elem);
    }

    std::unique_lock<std::mutex> lock(mtx_);
    if (max_new_alloc_size_ < 0 && new_alloc_size_ > buffers_.size() * 0.5 &&
        (elem < (T *)&buffers_.front() || elem > (T *)&buffers_.back())) {
      delete elem;
      elem = nullptr;
      new_alloc_size_--;
    } else {
      points_.push_back(elem);
    }
  }

  inline int Free() {
    std::unique_lock<std::mutex> lock(mtx_);

    if (buffers_.size() + new_alloc_size_ != points_.size()) {
      return 0;
    }
    for (T *&elem : points_) {
      if (buffers_.empty() || elem < (T *)&buffers_.front() ||
          elem > (T *)&buffers_.back()) {
        delete elem;
        elem = nullptr;
      }
    }

    buffers_.clear();
    points_.clear();
    new_alloc_size_ = 0;

    return 1;
  }

  std::vector<T, Alloc> buffers_;
  std::deque<T *> points_;
  uint32_t new_alloc_size_;
  int max_new_alloc_size_;

  CallBackFunc malloc_callback_;
  CallBackFunc acquire_callback_;
  CallBackFunc recycle_callback_;

  bool is_init_;
  std::mutex mtx_;
};

} // namespace dso
