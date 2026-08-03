#pragma once

#include "../camera_model/vio_def.h"
#include "algs_tools_allocated_buffer.h"

namespace dso {

class ImagesBuffer {
 public:
  using Ptr = typename BufferContainer<AlgsImage>::Ptr;

  inline static bool Initial(const int pre_alloc_size, const int width, const int height) {
    auto* static_container = Instance();
    if (!static_container->IsInitial()) {
      static_container->Initial(nullptr, nullptr, nullptr, pre_alloc_size, -1, width, height);
    }
    return static_container->IsInitial();
  }
  inline static void SetInitial(const bool flag) {
    auto* static_container = Instance();
    static_container->SetInitial(flag);
  };
  inline static void Release() {
    if (global_buffer_) {
      delete global_buffer_;
    }
    global_buffer_ = nullptr;
  }

  inline static Ptr Acquire(int width, int height) {
    auto* static_container = Instance();
    assert(static_container->IsInitial());
    return static_container->Acquire(width, height);
  }

  inline static uint32_t Size() { return Instance()->Size(); }

 private:
  static BufferContainer<AlgsImage>* Instance() {
    if (!global_buffer_) {
      global_buffer_ = new BufferContainer<AlgsImage>();
    }
    return global_buffer_;
  }

  static BufferContainer<AlgsImage>* global_buffer_;
};

}  // namespace dso