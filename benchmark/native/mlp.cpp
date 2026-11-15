#include "reflop.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

int main(int argc,char **argv) {
    constexpr int B=8,D=16,H=32,O=4;
    auto data=[](int n,float scale) {
        std::vector<float> v(n);
        for (int i=0;i<n;i++) v[i]=scale*std::sin(.37f*i);
        return v;
    };
    auto x=data(B*D,1),w1=data(H*D,.1f),b1=data(H,.02f),w2=data(O*H,.1f),b2=data(O,.02f);
    std::vector<double> hidden(B*H),reference(B*O);
    for (int i=0;i<B;i++) for (int j=0;j<H;j++) {
        double sum=b1[j];
        for (int k=0;k<D;k++) sum+=(double)x[i*D+k]*w1[j*D+k];
        hidden[i*H+j]=std::max(0.,sum);
    }
    for (int i=0;i<B;i++) for (int j=0;j<O;j++) {
        double sum=b2[j];
        for (int k=0;k<H;k++) sum+=hidden[i*H+k]*w2[j*H+k];
        reference[i*O+j]=sum;
    }
    try {
        reflop::Session s(nullptr,argc>1 ? argv[1] : "cuda");
        reflop::Graph g;
        auto a=g.input(REFLOP_F32,{B,D}),u=g.input(REFLOP_F32,{H,D}),b=g.input(REFLOP_F32,{H});
        auto v=g.input(REFLOP_F32,{O,H}),c=g.input(REFLOP_F32,{O});
        auto ut=g.op("aten.t.default").tensor(u).end();
        auto h=g.op("aten.addmm.default").tensor(b).tensor(a).tensor(ut).i(1).i(1).end();
        auto act=g.op("aten.relu.default").tensor(h).end();
        auto vt=g.op("aten.t.default").tensor(v).end();
        auto z=g.op("aten.addmm.default").tensor(c).tensor(act).tensor(vt).i(1).i(1).end();
        g.output(z,REFLOP_F32,{B,O});
        auto start=std::chrono::steady_clock::now();
        auto out=s.run(g,{s.upload(x,{B,D}),s.upload(w1,{H,D}),s.upload(b1,{H}),s.upload(w2,{O,H}),s.upload(b2,{O})});
        auto result=s.fetch<float>(out[0]);
        if (result.size()!=B*O) return 2;
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        double worst=0;
        for (int i=0;i<B*O;i++) worst=std::max(worst,std::abs(result[i]-reference[i]));
        std::printf("{\"max_abs_error\":%.9g,\"upload_register_execute_fetch_seconds\":%.9g}\n",worst,seconds);
        return worst<=.002 ? 0 : 2;
    } catch (const std::exception &e) {
        std::fprintf(stderr,"%s\n",e.what());
        return 1;
    }
}
