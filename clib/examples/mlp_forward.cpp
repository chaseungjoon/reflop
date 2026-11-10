// SPDX-License-Identifier: Apache-2.0
// Two-layer MLP forward on a reFLOP server with the C++ wrapper; checks
// against a local reference.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "reflop.hpp"

int main() {
  const int64_t B = 4, D = 8, H = 16, O = 3;
  auto fill = [](size_t n, float scale) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = scale * std::sin(0.37f * static_cast<float>(i) + 1.0f);
    return v;
  };
  auto x = fill(B * D, 1.0f), w1 = fill(H * D, 0.5f), b1 = fill(H, 0.1f), w2 = fill(O * H, 0.5f), b2 = fill(O, 0.1f);

  // Local reference: y = relu(x W1^T + b1) W2^T + b2
  std::vector<float> h(B * H), ref(B * O);
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < H; ++j) {
      float s = b1[j];
      for (int k = 0; k < D; ++k) s += x[i * D + k] * w1[j * D + k];
      h[i * H + j] = std::max(s, 0.0f);
    }
    for (int j = 0; j < O; ++j) {
      float s = b2[j];
      for (int k = 0; k < H; ++k) s += h[i * H + k] * w2[j * H + k];
      ref[i * O + j] = s;
    }
  }

  try {
    reflop::Session s;
    reflop::Graph g;
    auto gx = g.input(REFLOP_F32, {B, D});
    auto gw1 = g.input(REFLOP_F32, {H, D});
    auto gb1 = g.input(REFLOP_F32, {H});
    auto gw2 = g.input(REFLOP_F32, {O, H});
    auto gb2 = g.input(REFLOP_F32, {O});
    auto w1t = g.op("aten.t.default").tensor(gw1).end();
    // addmm(self, mat1, mat2, *, beta=1, alpha=1)
    auto a1 = g.op("aten.addmm.default").tensor(gb1).tensor(gx).tensor(w1t).i(1).i(1).end();
    auto r1 = g.op("aten.relu.default").tensor(a1).end();
    auto w2t = g.op("aten.t.default").tensor(gw2).end();
    auto y = g.op("aten.addmm.default").tensor(gb2).tensor(r1).tensor(w2t).i(1).i(1).end();
    g.output(y, REFLOP_F32, {B, O});

    auto out = s.run(g, {s.upload(x, {B, D}), s.upload(w1, {H, D}), s.upload(b1, {H}), s.upload(w2, {O, H}),
                         s.upload(b2, {O})});
    auto got = s.fetch<float>(out[0]);
    double err = 0;
    for (size_t i = 0; i < ref.size(); ++i) err = std::max(err, static_cast<double>(std::fabs(got[i] - ref[i])));
    std::printf("MLP forward (%lldx%lld -> %lld -> %lld) on the server: max abs error %.2e\n",
                static_cast<long long>(B), static_cast<long long>(D), static_cast<long long>(H),
                static_cast<long long>(O), err);
    return err < 1e-4 ? 0 : 2;
  } catch (const reflop::Error &e) {
    std::fprintf(stderr, "reflop: %s\n", e.what());
    return 1;
  }
}
