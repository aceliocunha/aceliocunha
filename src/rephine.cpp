#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include "unionfind.hh"

// ===== compute_rephine_raw corrigida =====
template <typename float_t, typename int_t>
void compute_rephine_raw(
    torch::TensorAccessor<float_t, 1> filtered_v,
    torch::TensorAccessor<float_t, 1> filtered_e,
    torch::TensorAccessor<int_t, 2> edge_index,
    torch::TensorAccessor<int_t, 1> parents,
    torch::TensorAccessor<int_t, 1> sorting_space,
    torch::TensorAccessor<int_t, 2> pers_indices,
    torch::TensorAccessor<int_t, 2> pers1_indices,
    int_t vertex_begin, int_t vertex_end,
    int_t edge_begin, int_t edge_end)
{
  auto n_edges = edge_end - edge_begin;
  int_t* sorting_begin = sorting_space.data() + edge_begin;
  int_t* sorting_end = sorting_space.data() + edge_end;
  std::stable_sort(sorting_begin, sorting_end, [&filtered_e](int_t i, int_t j) {
    return filtered_e[i] < filtered_e[j];
  });

  for (int_t i = 0; i < n_edges; ++i) {
    int_t cur_edge_index = sorting_space[edge_begin + i];
    int_t u = edge_index[cur_edge_index][0];
    int_t v = edge_index[cur_edge_index][1];

    int_t root_u = UnionFind<int_t>::find(parents, u);
    int_t root_v = UnionFind<int_t>::find(parents, v);

    if (root_u == root_v) {
      pers1_indices[cur_edge_index][0] = cur_edge_index;
      pers1_indices[cur_edge_index][1] = cur_edge_index;
      continue;
    }

    float_t birth_u = filtered_v[root_u];
    float_t birth_v = filtered_v[root_v];

    int_t younger = (birth_u < birth_v) ? root_u : root_v;
    int_t older = (birth_u < birth_v) ? root_v : root_u;

    pers_indices[younger][0] = cur_edge_index;
    UnionFind<int_t>::merge(parents, younger, older);
  }

  for (int_t i = vertex_begin; i < vertex_end; ++i) {
    if (parents[i] == i) {
      pers_indices[i][0] = -1;
    }
  }
}


template <typename float_t, typename int_t>
void compute_rephine_ptrs(
    torch::TensorAccessor<float_t, 2> filtered_v,
    torch::TensorAccessor<float_t, 2> filtered_e,
    torch::TensorAccessor<int_t, 2> edge_index,
    torch::TensorAccessor<int_t, 1> vertex_slices,
    torch::TensorAccessor<int_t, 1> edge_slices,
    torch::TensorAccessor<int_t, 2> parents,
    torch::TensorAccessor<int_t, 2> sorting_space,
    torch::TensorAccessor<int_t, 3> pers_ind,
    torch::TensorAccessor<int_t, 3> pers1_ind)
{
  auto n_graphs = vertex_slices.size(0) - 1;
  auto n_filtrations = filtered_v.size(0);

  at::parallel_for(0, n_graphs * n_filtrations, 0, [&](int64_t begin, int64_t end) {
    for (auto i = begin; i < end; i++) {
      auto instance = i / n_filtrations;
      auto filtration = i % n_filtrations;
      compute_rephine_raw<float_t, int_t>(
          filtered_v[filtration], filtered_e[filtration], edge_index,
          parents[filtration], sorting_space[filtration],
          pers_ind[filtration], pers1_ind[filtration],
          vertex_slices[instance], vertex_slices[instance + 1],
          edge_slices[instance], edge_slices[instance + 1]);
    }
  });
}

// ===== (batched, multithread) =====
std::tuple<torch::Tensor, torch::Tensor>
compute_rephine_batched_mt(torch::Tensor filtered_v,
                           torch::Tensor filtered_e,
                           torch::Tensor edge_index,
                           torch::Tensor vertex_slices,
                           torch::Tensor edge_slices)
{
  const int64_t n_nodes = filtered_v.size(1);
  const int64_t n_edges = filtered_e.size(1);
  const int64_t n_filtrations = filtered_v.size(0);

  auto iopts = edge_index.options().dtype(torch::kInt64).requires_grad(false);
  auto pers_ind = torch::full({n_filtrations, n_nodes, 2}, -1, iopts);
  auto pers1_ind = torch::full({n_filtrations, n_edges, 2}, -1, iopts);

  auto parents = torch::arange(n_nodes, iopts).unsqueeze(0).repeat({n_filtrations, 1});
  auto sorting_space = torch::arange(n_edges, iopts).unsqueeze(0).repeat({n_filtrations, 1}).contiguous();

  AT_DISPATCH_FLOATING_TYPES(filtered_v.scalar_type(), "rephine", ([&] {
    using float_t = scalar_t;
    AT_DISPATCH_INTEGRAL_TYPES(edge_index.scalar_type(), "rephine", ([&] {
      using int_t = scalar_t;
      compute_rephine_ptrs<float_t, int_t>(
          filtered_v.accessor<float_t, 2>(),
          filtered_e.accessor<float_t, 2>(),
          edge_index.accessor<int_t, 2>(),
          vertex_slices.accessor<int_t, 1>(),
          edge_slices.accessor<int_t, 1>(),
          parents.accessor<int_t, 2>(),
          sorting_space.accessor<int_t, 2>(),
          pers_ind.accessor<int_t, 3>(),
          pers1_ind.accessor<int_t, 3>());
    }));
  }));

  auto pers_v = filtered_v.unsqueeze(2);
  auto pers_e = filtered_e.unsqueeze(2);

  return std::make_tuple(pers_ind.to(torch::kLong), pers1_ind.to(torch::kLong));
}

// ===== PYBIND11 =====
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_rephine_batched_mt", &compute_rephine_batched_mt,
        py::call_guard<py::gil_scoped_release>(),
        "Persistence routine multi-threaded");
}
