#include "reflop.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK(call) do { if ((call) != 0) { fprintf(stderr, "%s: %s\n", #call, reflop_last_error()); exit(1); } } while (0)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    enum { M=64, K=48, N=32, RUNS=12 };
    float a[M*K], b[K*N], output[M*N];
    double reference[M*N], samples[RUNS], worst=0;
    for (int i=0;i<M*K;i++) a[i]=(float)((i*7)%13)/13-.5f;
    for (int i=0;i<K*N;i++) b[i]=(float)((i*5)%11)/11-.5f;
    for (int i=0;i<M;i++) for (int j=0;j<N;j++) {
        double sum=0;
        for (int k=0;k<K;k++) sum+=(double)a[i*K+k]*b[k*N+j];
        reference[i*N+j]=sum;
    }
    reflop_session s;
    CHECK(reflop_connect(NULL,argc>1 ? argv[1] : "cuda",&s));
    int64_t sa[2]={M,K},sb[2]={K,N},so[2]={M,N};
    reflop_handle ha,hb;
    CHECK(reflop_upload(s,REFLOP_F32,2,sa,a,sizeof(a),&ha));
    CHECK(reflop_upload(s,REFLOP_F32,2,sb,b,sizeof(b),&hb));
    reflop_graph g=reflop_graph_new();
    reflop_value x=reflop_graph_input(g,REFLOP_F32,2,sa);
    reflop_value y=reflop_graph_input(g,REFLOP_F32,2,sb),z;
    CHECK(reflop_op_begin(g,"aten.mm.default"));
    CHECK(reflop_arg_tensor(g,x)); CHECK(reflop_arg_tensor(g,y)); CHECK(reflop_op_end(g,&z));
    CHECK(reflop_graph_output(g,z,REFLOP_F32,2,so));
    reflop_graph_id id;
    CHECK(reflop_register(s,g,&id));
    reflop_handle in[2]={ha,hb},out[1];
    for (int i=0;i<RUNS;i++) {
        double start=now();
        CHECK(reflop_execute(s,id,in,2,out,1));
        size_t bytes=0;
        CHECK(reflop_fetch(s,out[0],output,sizeof(output),&bytes));
        samples[i]=now()-start;
        if (bytes!=sizeof(output)) return 2;
        for (int j=0;j<M*N;j++) worst=fmax(worst,fabs(output[j]-reference[j]));
        CHECK(reflop_free(s,out[0]));
    }
    printf("{\"max_abs_error\":%.9g,\"execute_fetch_seconds\":[",worst);
    for (int i=0;i<RUNS;i++) printf("%s%.9g",i ? "," : "",samples[i]);
    printf("]}\n");
    reflop_graph_free(g);
    CHECK(reflop_close(s));
    return worst<=.002 ? 0 : 2;
}
