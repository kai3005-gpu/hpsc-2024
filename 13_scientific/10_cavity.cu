#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
typedef vector<vector<float>> matrix;

__global__ void updateVariable(float *b, float *u, float *v, float dx, float dy, int nx, int ny, float rho, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nx * ny && idx > nx) {
        b[idx] = rho * (1 / dt * 
                        ((u[idx + 1] - u[idx - 1]) / (2 * dx) + (v[idx + nx] - v[idx - nx]) / (2 * dy)) -
                        pow((u[idx + 1] - u[idx - 1]) / (2 * dx), 2) -
                        2 * ((u[idx - nx] - u[idx - nx]) / (2 * dy) *
                        (v[idx + 1] - v[idx - 1]) / (2 * dx)) -
                        pow((v[idx + nx] - v[idx - nx]) / (2 * dy), 2));
    }
}

__global__ void updatePressure(float *p, float *pn, float *b, float dx, float dy, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nx * ny && idx > nx) {
        p[idx] = ((pn[idx + 1] + pn[idx - 1]) * dy * dy +
                 (pn[idx + nx] + pn[idx - nx]) * dx * dx -
                 b[idx] * dx * dx * dy * dy) /
                 (2 * (dx * dx + dy * dy));
    }
}

__global__ void applyBoundaryConditionsP(float *p, int nx, int ny) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < nx) {
        p[idx] = p[nx + idx]; // p[0][i] = p[1][i]
        p[(ny - 1) * nx + idx] = 0; // p[ny - 1][i] = 0
    }
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idy < ny) {
        p[idy * nx] = p[idy * nx + 1]; // p[j][0] = p[j][1]
        p[idy * nx + nx - 1] = p[idy * nx + nx - 2]; // p[j][nx - 1] = p[j][nx - 2]
    }
}

__global__ void updateUV(float *u, float *un, float *v, float *vn, float *p, int nx, int ny, float dt, float dx, float dy, float rho, float nu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (idx < nx * ny && idx > nx) {
        u[idx] = un[idx] - un[idx] * dt * (un[idx] - un[idx - 1]) / dx 
                         - vn[idx] * dt * (un[idx] - un[idx - nx]) / dy
                         - dt * (p[idx + 1] - p[idx - 1]) / (2 * rho * dx)
                         + nu * dt * (un[idx + 1] - 2 * un[idx] + un[idx - 1]) / (dx * dx)
                         + nu * dt * (un[idx + nx] - 2 * un[idx] + un[idx - nx]) / (dy * dy);

        v[idx] = vn[idx] - un[idx] * dt * (vn[idx] - vn[idx - 1]) / dx
                         - vn[idx] * dt * (vn[idx] - vn[idx - nx]) / dy 
                         - dt * (p[idx + nx] - p[idx - nx]) / (2 * rho * dy)
                         + nu * dt * (vn[idx + 1] - 2 * vn[idx] + vn[idx - 1]) / (dx * dx) 
                         + nu * dt * (vn[idx + nx] - 2 * vn[idx] + vn[idx - nx]) / (dy * dy);
    }
}

__global__ void applyBoundaryConditionsUV(float *u, float *v, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nx) {
        // 下部境界
        u[idx] = 0;
        v[idx] = 0;

        // 上部境界
        int idx_top = (ny - 1) * nx + idx;
        u[idx_top] = 1;
        v[idx_top] = 0;
    }

    if (idx < ny) {
        // 左端
        int idx_left = idx * nx;
        u[idx_left] = 0;
        v[idx_left] = 0;

        // 右端
        int idx_right = idx * nx + (nx - 1);
        u[idx_right] = 0;
        v[idx_right] = 0;
    }
}

int main(){
    const int nx = 41;
    const int ny = 41;
    int nt = 3000;
    int nit = 300;
    float dx = 2. / (nx - 1);
    float dy = 2. / (ny - 1);
    float dt = 0.01;
    float rho = 1.;
    float nu = 0.02;
    vector<float> x(nx);
    vector<float> y(ny);
    for (int i = 0; i < nx; i++){
        x[i] = i * dx;
    }
    for (int i = 0; i < ny; i++){
        y[i] = i * dy;
    }

    // 1次元配列として定義
    std::vector<float> u(ny * nx);
    std::vector<float> v(ny * nx);
    std::vector<float> p(ny * nx);
    std::vector<float> b(ny * nx);

    // 全ての要素を0に初期化）
    for (int i = 0; i < ny * nx; ++i) {
        u[i] = 0.0f;
        v[i] = 0.0f;
        p[i] = 0.0f;
        b[i] = 0.0f;
    }

    //ブロックあたりのスレッド数（blocksize)を1024、
    //ブロックの総数（gridsize）を(N + 1024 - 1)/1024用意する
    const int blocksize = 1024;
    dim3 block(blocksize, 1, 1);

    // CUDA memory allocation
    float *dev_u, *dev_v, *dev_un, *dev_vn, *dev_p, *dev_pn, *dev_b;
    cudaMallocManaged(&dev_u, nx * ny * sizeof(float));
    cudaMallocManaged(&dev_v, nx * ny * sizeof(float));
    cudaMallocManaged(&dev_un, nx * ny * sizeof(float));
    cudaMallocManaged(&dev_vn, nx * ny * sizeof(float));
    cudaMallocManaged(&dev_p, nx * ny * sizeof(float));
    cudaMallocManaged(&dev_pn, nx * ny * sizeof(float));
    cudaMallocManaged(&dev_b, nx * ny * sizeof(float));

    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    for (int n = 0; n < nt; n++){
        const int step = (nx - 2) * (ny - 2);
        dim3 grid2d((step + block.x - 1) / block.x, 1, 1);
        dim3 grid1d((nx + block.x - 2) / block.x, 1, 1);

	    auto t1 = chrono::steady_clock::now();
        // u,vのデータをデバイスに送信
        cudaMemcpy(dev_u, u.data(), nx * ny * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_v, v.data(), nx * ny * sizeof(float), cudaMemcpyHostToDevice);
        // bの計算
        updateVariable<<<grid2d, block>>>(dev_b, dev_u, dev_v, dx, dy, nx, ny, rho, dt);
        cudaDeviceSynchronize();
        // bの計算時間
        auto t2 = chrono::steady_clock::now();
        double time1 = chrono::duration<double>(t2 - t1).count();
        printf("step_b=%d: %lf seconds\n", n, time1);
        // デバイスメモリからホストメモリにデータをコピー
        cudaMemcpy(b.data(), dev_b, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

        // p,bのデータをデバイスに送信
        cudaMemcpy(dev_pn, p.data(), nx * ny * sizeof(float), cudaMemcpyHostToDevice);
        // pの計算
        for (int it = 0; it < nit; it++){
            updatePressure<<<grid2d, block>>>(dev_p, dev_pn, dev_b, dx, dy, nx, ny);
            cudaDeviceSynchronize();
            // 境界条件
            applyBoundaryConditionsP<<<grid1d, block>>>(dev_p, nx, ny);
            cudaDeviceSynchronize();
        }
        // pの計算時間
        auto t3 = chrono::steady_clock::now();
        double time2 = chrono::duration<double>(t3 - t2).count();
        printf("step_p=%d: %lf seconds\n", n, time2);
        // デバイスメモリからホストメモリにデータをコピー
        cudaMemcpy(p.data(), dev_p, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

        // u,vのコピー
        cudaMemcpy(dev_un, dev_u, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_vn, dev_v, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
        // u,vの計算
        updateUV<<<grid2d, blocksize>>>(dev_u, dev_un, dev_v, dev_vn, dev_p, nx, ny, dt, dx, dy, rho, nu);
        cudaDeviceSynchronize();
        // 境界条件の適用
        applyBoundaryConditionsUV<<<grid1d, blocksize>>>(dev_u, dev_v, nx, ny);
        cudaDeviceSynchronize();
        // u,vの計算時間
        auto t4 = chrono::steady_clock::now();
        double time3 = chrono::duration<double>(t4 - t3).count();
        printf("step_u=%d: %lf seconds\n", n, time3);
        // デバイスメモリからホストメモリにデータをコピー
        cudaMemcpy(u.data(), dev_u, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(v.data(), dev_v, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

        // Write data
        if (n % 10 == 0) {
            for (int j=0; j<ny; j++) {
                for (int i=0; i<nx; i++) {
                    int idx = j * nx + i;
                    ufile << u[idx] << " ";
                    vfile << v[idx] << " ";
                    pfile << p[idx] << " ";
                }
            }
            ufile << "\n";
            vfile << "\n";
            pfile << "\n";
        }
    }

    ufile.close();
    vfile.close();
    pfile.close();

    cudaFree(dev_p);
    cudaFree(dev_pn);
    cudaFree(dev_b);

    return 0;
}
