#pragma once
#include <ATen/ATen.h>

template <typename int_t>
class UnionFind {
public:
  static int_t find(at::TensorAccessor<int_t,1> p, int_t u) {
    auto parent = p[u];
    if (parent == u) return u;
    p[u] = find(p, parent);     
    return p[u];
  }
  static void merge(at::TensorAccessor<int_t,1> p, int_t u, int_t v) {
    auto ru = find(p,u), rv = find(p,v);
    if (ru != rv) p[ru] = rv;   
  }
};
