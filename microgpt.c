/* A dependency-free, scalar C translation of microgpt.py. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef struct { double d, g, l1, l2; int c1, c2, n; } Node;
typedef struct { Node *a; size_t n, cap; } Graph;
static int nn(Graph *g,double d,int c1,double l1,int c2,double l2,int n){if(g->n==g->cap){g->cap=g->cap?g->cap*2:1024;g->a=realloc(g->a,g->cap*sizeof(Node));}int i=(int)g->n++;g->a[i]=(Node){d,0,l1,l2,c1,c2,n};return i;}
static int cn(Graph*g,double d){return nn(g,d,0,0,0,0,0);}
static int add(Graph*g,int a,int b){return nn(g,g->a[a].d+g->a[b].d,a,1,b,1,2);}
static int mul(Graph*g,int a,int b){return nn(g,g->a[a].d*g->a[b].d,a,g->a[b].d,b,g->a[a].d,2);}
static int mulf(Graph*g,int a,double x){int b=cn(g,x);return mul(g,a,b);}
static int negv(Graph*g,int a){return mulf(g,a,-1);}
static int powv(Graph*g,int a,double p){return nn(g,pow(g->a[a].d,p),a,p*pow(g->a[a].d,p-1),0,0,1);}
static int logv(Graph*g,int a){return nn(g,log(g->a[a].d),a,1/g->a[a].d,0,0,1);}
static int expv(Graph*g,int a){double d=exp(g->a[a].d);return nn(g,d,a,d,0,0,1);}
static int relu(Graph*g,int a){return nn(g,fmax(0,g->a[a].d),a,g->a[a].d>0,0,0,1);}
static int sumv(Graph*g,int*n,int len){int z=cn(g,0);for(int i=0;i<len;i++)z=add(g,z,n[i]);return z;}

typedef struct {uint32_t mt[624];int ix;double spare;int has;} RNG;
static void seed(RNG*r,uint32_t s){r->mt[0]=19650218;r->ix=624;r->has=0;for(int i=1;i<624;i++)r->mt[i]=1812433253u*(r->mt[i-1]^(r->mt[i-1]>>30))+i;int i=1,j=0,k=624;while(k--){r->mt[i]=(r->mt[i]^((r->mt[i-1]^(r->mt[i-1]>>30))*1664525u))+s+j;i++;j++;if(i>=624){r->mt[0]=r->mt[623];i=1;}if(j>=1)j=0;}k=623;while(k--){r->mt[i]=(r->mt[i]^((r->mt[i-1]^(r->mt[i-1]>>30))*1566083941u))-i;i++;if(i>=624){r->mt[0]=r->mt[623];i=1;}}r->mt[0]=0x80000000;}
static uint32_t u32(RNG*r){if(r->ix>=624){for(int i=0;i<624;i++){uint32_t y=(r->mt[i]&0x80000000)|(r->mt[(i+1)%624]&0x7fffffff);r->mt[i]=r->mt[(i+397)%624]^(y>>1)^((y&1)?0x9908b0df:0);}r->ix=0;}uint32_t y=r->mt[r->ix++];y^=y>>11;y^=(y<<7)&0x9d2c5680;y^=(y<<15)&0xefc60000;y^=y>>18;return y;}
static double rnd(RNG*r){return ((double)(u32(r)>>5)*67108864+(u32(r)>>6))/9007199254740992.0;}
static int below(RNG*r,int n){int k=0,x=n;while(x){k++;x>>=1;}do{x=(int)(u32(r)>>(32-k));}while(x>=n);return x;}
static double gauss(RNG*r){if(r->has){r->has=0;return r->spare;}double a=rnd(r)*2*M_PI,b=sqrt(-2*log(1-rnd(r)));r->spare=sin(a)*b;r->has=1;return cos(a)*b;}
static void shuffle(RNG*r,char**a,int n){for(int i=n-1;i>0;i--){int j=below(r,i+1);char*t=a[i];a[i]=a[j];a[j]=t;}}
static int choice(RNG*r,double*w,int n){double total=0,x;for(int i=0;i<n;i++)total+=w[i];x=rnd(r)*total;for(int i=0;i<n;i++)if((x-=w[i])<0)return i;return n-1;}

static int *matrix(Graph*g,int rows,int cols,RNG*r){int*m=malloc((size_t)rows*cols*sizeof(int));for(int i=0;i<rows*cols;i++)m[i]=cn(g,.08*gauss(r));return m;}
static int *linear(Graph*g,int*x,int nx,int*w,int rows,int cols){int*y=malloc(rows*sizeof(int));for(int i=0;i<rows;i++){int*t=malloc(cols*sizeof(int));for(int j=0;j<cols;j++)t[j]=mul(g,w[i*cols+j],x[j]);y[i]=sumv(g,t,cols);free(t);}return y;}
static int *softmax(Graph*g,int*x,int n){double mx=-INFINITY;for(int i=0;i<n;i++)if(g->a[x[i]].d>mx)mx=g->a[x[i]].d;int*e=malloc(n*sizeof(int));for(int i=0;i<n;i++){int c=cn(g,-mx);e[i]=expv(g,add(g,x[i],c));}int total=sumv(g,e,n),inv=powv(g,total,-1);for(int i=0;i<n;i++)e[i]=mul(g,e[i],inv);return e;}
static int *rms(Graph*g,int*x,int n){int*t=malloc(n*sizeof(int));for(int i=0;i<n;i++)t[i]=mul(g,x[i],x[i]);int ms=mulf(g,sumv(g,t,n),1.0/n),c=cn(g,1e-5),scale=powv(g,add(g,ms,c),-.5);for(int i=0;i<n;i++)t[i]=mul(g,x[i],scale);return t;}

static int *gpt(Graph*g,int tok,int pos,int keys[][256],int vals[][256],int *kn,int **s){
  int x[16];for(int i=0;i<16;i++)x[i]=add(g,s[0][tok*16+i],s[1][pos*16+i]);int*n=rms(g,x,16);memcpy(x,n,16*sizeof(int));free(n);
  for(int li=0;li<1;li++){int res[16];memcpy(res,x,sizeof res);n=rms(g,x,16);int*q=linear(g,n,16,s[3],16,16),*k=linear(g,n,16,s[4],16,16),*v=linear(g,n,16,s[5],16,16);free(n);memcpy(keys[li]+kn[li]*16,k,16*sizeof(int));memcpy(vals[li]+kn[li]*16,v,16*sizeof(int));kn[li]++;free(k);free(v);int att[16],ai=0;for(int h=0;h<4;h++){int scores[16];for(int t=0;t<kn[li];t++){int dots[4];for(int j=0;j<4;j++)dots[j]=mul(g,q[h*4+j],keys[li][t*16+h*4+j]);scores[t]=mulf(g,sumv(g,dots,4),.5);}int*aw=softmax(g,scores,kn[li]);for(int j=0;j<4;j++){int terms[16];for(int t=0;t<kn[li];t++)terms[t]=mul(g,aw[t],vals[li][t*16+h*4+j]);att[ai++]=sumv(g,terms,kn[li]);}free(aw);}int*y=linear(g,att,16,s[6],16,16);for(int i=0;i<16;i++)x[i]=add(g,y[i],res[i]);free(y);memcpy(res,x,sizeof res);n=rms(g,x,16);y=linear(g,n,16,s[7],64,16);free(n);for(int i=0;i<64;i++)y[i]=relu(g,y[i]);n=linear(g,y,64,s[8],16,64);free(y);for(int i=0;i<16;i++)x[i]=add(g,n[i],res[i]);free(n);free(q);}
  return linear(g,x,16,s[2],27,16);
}
static void backward(Graph*g,int root){int*N=malloc(g->n*sizeof(int)),*top=malloc(g->n*sizeof(int));char*seen=calloc(g->n,1);size_t sp=0,tn=0;N[sp++]=root;while(sp){int v=N[--sp];if(v<0){top[tn++]=-v-1;continue;}if(seen[v])continue;seen[v]=1;N[sp++]=-v-1;if(g->a[v].n==2)N[sp++]=g->a[v].c2;if(g->a[v].n>=1)N[sp++]=g->a[v].c1;}g->a[root].g=1;while(tn){int v=top[--tn];if(g->a[v].n>=1)g->a[g->a[v].c1].g+=g->a[v].l1*g->a[v].g;if(g->a[v].n==2)g->a[g->a[v].c2].g+=g->a[v].l2*g->a[v].g;}free(N);free(top);free(seen);}

int main(void){RNG r;seed(&r,42);FILE*f=fopen("input.txt","r");if(!f)return 1;char**docs=NULL;int nd=0,cap=0;char*line=NULL;size_t z=0;while(getline(&line,&z,f)>0){char*a=line;while(*a&&(*a==' '||*a=='\t'||*a=='\n'||*a=='\r'))a++;char*e=a+strlen(a);while(e>a&&(e[-1]==' '||e[-1]=='\t'||e[-1]=='\n'||e[-1]=='\r'))*--e=0;if(*a){if(nd==cap){cap=cap?cap*2:1024;docs=realloc(docs,cap*sizeof(char*));}docs[nd++]=strdup(a);}}free(line);fclose(f);shuffle(&r,docs,nd);printf("num docs: %d\n",nd);char chars[256];int nc=0;for(int i=0;i<nd;i++)for(char*a=docs[i];*a;a++){int seen=0;for(int j=0;j<nc;j++)if(chars[j]==*a)seen=1;if(!seen)chars[nc++]=*a;}for(int i=0;i<nc;i++)for(int j=i+1;j<nc;j++)if(chars[j]<chars[i]){char t=chars[i];chars[i]=chars[j];chars[j]=t;}int bos=nc,vocab=nc+1;printf("vocab size: %d\n",vocab);Graph g={0};int*s[9];s[0]=matrix(&g,vocab,16,&r);s[1]=matrix(&g,16,16,&r);s[2]=matrix(&g,vocab,16,&r);s[3]=matrix(&g,16,16,&r);s[4]=matrix(&g,16,16,&r);s[5]=matrix(&g,16,16,&r);s[6]=matrix(&g,16,16,&r);s[7]=matrix(&g,64,16,&r);s[8]=matrix(&g,16,64,&r);size_t pc=g.n;printf("num params: %zu\n",pc);double*m=calloc(pc,sizeof(double)),*vv=calloc(pc,sizeof(double));int steps=1000;char*env=getenv("MICROGPT_STEPS");if(env)steps=atoi(env);
  for(int step=0;step<steps;step++){int tok[17],len=strlen(docs[step%nd]);tok[0]=bos;for(int i=0;i<len;i++){for(int j=0;j<nc;j++)if(chars[j]==docs[step%nd][i])tok[i+1]=j;}tok[len+1]=bos;int n=len+1;if(n>16)n=16;int keys[1][256]={0},vals[1][256]={0},kn[1]={0},losses[32],nl=0;for(int p=0;p<n;p++){int*lo=gpt(&g,tok[p],p,keys,vals,kn,s);int*pr=softmax(&g,lo,vocab),lp=logv(&g,pr[tok[p+1]]);losses[nl++]=negv(&g,lp);free(lo);free(pr);}int loss=mulf(&g,sumv(&g,losses,nl),1.0/nl);backward(&g,loss);double lr=.01*(1-(double)step/steps);for(size_t i=0;i<pc;i++){double grad=g.a[i].g;m[i]=.85*m[i]+.15*grad;vv[i]=.99*vv[i]+.01*grad*grad;double mh=m[i]/(1-pow(.85,step+1)),vh=vv[i]/(1-pow(.99,step+1));g.a[i].d-=lr*mh/(sqrt(vh)+1e-8);g.a[i].g=0;}printf("step %4d / %4d | loss %.4f\r",step+1,steps,g.a[loss].d);g.n=pc;}
  puts("\n--- inference (new, hallucinated names) ---");for(int si=0;si<20;si++){int keys[1][256]={0},vals[1][256]={0},kn[1]={0},tok=bos;char sample[32]={0};int sl=0;for(int p=0;p<16;p++){int*lo=gpt(&g,tok,p,keys,vals,kn,s),*sc=malloc(vocab*sizeof(int));for(int i=0;i<vocab;i++)sc[i]=mulf(&g,lo[i],2);int*pr=softmax(&g,sc,vocab);double*w=malloc(vocab*sizeof(double));for(int i=0;i<vocab;i++)w[i]=g.a[pr[i]].d;tok=choice(&r,w,vocab);free(w);free(pr);free(sc);free(lo);if(tok==bos)break;sample[sl++]=chars[tok];}printf("sample %2d: %s\n",si+1,sample);}
  for(int i=0;i<nd;i++)free(docs[i]);free(docs);free(m);free(vv);free(g.a);return 0;
}
