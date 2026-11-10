// SPDX-License-Identifier: Apache-2.0
//
// reflop.hpp: C++17 RAII wrapper over the reFLOP C API (reflop.h).
//
//   reflop::Session s;                                   // default paired server
//   auto a = s.upload(std::vector<float>{1, 2, 3, 4}, {2, 2});
//   reflop::Graph g;
//   auto x = g.input(REFLOP_F32, {2, 2});
//   auto y = g.op("aten.mm.default").tensor(x).tensor(x).end();
//   g.output(y, REFLOP_F32, {2, 2});
//   auto out = s.run(g, {a});
//   std::vector<float> r = s.fetch<float>(out[0]);
#pragma once

#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "reflop.h"

namespace reflop {

struct Error : std::runtime_error {
  explicit Error(const std::string &m) : std::runtime_error(m) {}
};

inline void check(int rc) {
  if (rc != 0) throw Error(reflop_last_error());
}

template <typename T> constexpr reflop_dtype dtype_of();
template <> constexpr reflop_dtype dtype_of<float>() { return REFLOP_F32; }
template <> constexpr reflop_dtype dtype_of<double>() { return REFLOP_F64; }
template <> constexpr reflop_dtype dtype_of<int64_t>() { return REFLOP_I64; }
template <> constexpr reflop_dtype dtype_of<int32_t>() { return REFLOP_I32; }
template <> constexpr reflop_dtype dtype_of<uint8_t>() { return REFLOP_U8; }

class Graph;

class OpBuilder {
 public:
  OpBuilder(reflop_graph g, const char *op) : g_(g) { check(reflop_op_begin(g, op)); }
  OpBuilder &tensor(reflop_value v) { check(reflop_arg_tensor(g_, v)); return *this; }
  OpBuilder &tensors(const std::vector<reflop_value> &v) {
    check(reflop_arg_tensors(g_, v.data(), static_cast<int>(v.size())));
    return *this;
  }
  OpBuilder &i(int64_t v) { check(reflop_arg_int(g_, v)); return *this; }
  OpBuilder &ints(const std::vector<int64_t> &v) {
    check(reflop_arg_ints(g_, v.data(), static_cast<int>(v.size())));
    return *this;
  }
  OpBuilder &f(double v) { check(reflop_arg_float(g_, v)); return *this; }
  OpBuilder &b(bool v) { check(reflop_arg_bool(g_, v ? 1 : 0)); return *this; }
  OpBuilder &dtype(reflop_dtype v) { check(reflop_arg_dtype(g_, v)); return *this; }
  OpBuilder &enum_(const char *v) { check(reflop_arg_enum(g_, v)); return *this; }
  OpBuilder &device() { check(reflop_arg_device(g_)); return *this; }
  OpBuilder &none() { check(reflop_arg_none(g_)); return *this; }
  reflop_value end() {
    reflop_value v{};
    check(reflop_op_end(g_, &v));
    return v;
  }

 private:
  reflop_graph g_;
};

class Graph {
 public:
  Graph() : g_(reflop_graph_new()) {}
  ~Graph() { if (g_) reflop_graph_free(g_); }
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;
  Graph(Graph &&o) noexcept : g_(std::exchange(o.g_, 0)) {}

  reflop_value input(reflop_dtype dt, std::vector<int64_t> shape) {
    reflop_value v = reflop_graph_input(g_, dt, static_cast<int>(shape.size()), shape.data());
    if (v.node < -1) throw Error(reflop_last_error());
    ++inputs_;
    return v;
  }
  OpBuilder op(const char *name) { return OpBuilder(g_, name); }
  void output(reflop_value v, reflop_dtype dt, std::vector<int64_t> shape) {
    check(reflop_graph_output(g_, v, dt, static_cast<int>(shape.size()), shape.data()));
    ++outputs_;
  }
  reflop_graph raw() const { return g_; }
  int inputs() const { return inputs_; }
  int outputs() const { return outputs_; }

 private:
  reflop_graph g_;
  int inputs_ = 0, outputs_ = 0;
};

class Session {
 public:
  explicit Session(const char *server = nullptr, const char *device = "auto") {
    check(reflop_connect(server, device, &s_));
  }
  ~Session() { if (s_) reflop_close(s_); }
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  template <typename T>
  reflop_handle upload(const std::vector<T> &data, std::vector<int64_t> shape) {
    reflop_handle h{};
    check(reflop_upload(s_, dtype_of<T>(), static_cast<int>(shape.size()), shape.data(), data.data(),
                        data.size() * sizeof(T), &h));
    return h;
  }

  template <typename T> std::vector<T> fetch(reflop_handle h) {
    size_t n = 0;
    check(reflop_fetch(s_, h, nullptr, 0, &n));
    std::vector<T> out(n / sizeof(T));
    check(reflop_fetch(s_, h, out.data(), n, &n));
    return out;
  }

  void free(reflop_handle h) { check(reflop_free(s_, h)); }

  reflop_graph_id compile(const Graph &g) {
    reflop_graph_id id{};
    check(reflop_register(s_, g.raw(), &id));
    return id;
  }

  std::vector<reflop_handle> run(const Graph &g, const std::vector<reflop_handle> &inputs) {
    reflop_graph_id id = compile(g);
    std::vector<reflop_handle> out(static_cast<size_t>(g.outputs()));
    check(reflop_execute(s_, id, inputs.data(), static_cast<int>(inputs.size()), out.data(),
                         static_cast<int>(out.size())));
    return out;
  }

 private:
  reflop_session s_ = 0;
};

}  // namespace reflop
