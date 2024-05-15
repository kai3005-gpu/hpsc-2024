#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// デバイス側のバケットソートカーネル
__global__ void bucket_sort_kernel(int *key, int *bucket, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // スレッド番号
    if (i < n) {
        atomicAdd(&bucket[key[i]], 1);
        //printf("Thread %d: key[%d] = %d, bucket[%d] = %d\n", i, i, key[i], key[i], bucket[key[i]]);
    }
}

// バケットの累積和を計算するカーネル
__global__ void scan_kernel(int *a, int *b, int range) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= range) return;
    b[i] = a[i];
    __syncthreads();
    for (int j = 1; j < range; j <<= 1) {
        if (i >= j) {
            a[i] += b[i - j];
        }
        __syncthreads();
        b[i] = a[i];
        __syncthreads();
    }
    //printf("Thread %d: scan[%d] = %d\n", i, i, a[i]);
}

// デバイス側の結果を集めるカーネル
__global__ void gather_kernel(int *key, int *bucket, int range, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    for (int j = 0; j < range; j++) {
        // key[i]に格納される値がiとバケットの累積和との比較によって決定される
        if (i <= bucket[j]) {
            key[i] = j;
            //printf("Thread %d: key[%d] = %d, scan[%d] = %d\n", i, i, key[i], j, bucket[j]);
            break;
        }
    }
}

// ホスト側の関数
void bucket_sort_gpu(std::vector<int>& key, int range, int n) {
    int *d_key, *d_bucket, *d_bucket2;
    const int thread = 1024;

    // デバイスメモリの確保
    cudaMallocManaged(&d_key, n * sizeof(int));
    cudaMallocManaged(&d_bucket, range * sizeof(int));
    cudaMallocManaged(&d_bucket2, range * sizeof(int));

    // データの転送とバケットの初期化
    for (int i = 0; i < n; ++i) {
        d_key[i] = key[i];
    }
    for (int i = 0; i < range; ++i) {
        d_bucket[i] = 0;
        d_bucket2[i] = 0;
    }

    // バケットにデータを蓄積する
    bucket_sort_kernel<<<(n + thread - 1) / thread, thread>>>(d_key, d_bucket, n);
    cudaDeviceSynchronize();

    // バケットの累積和を計算
    scan_kernel<<<(range + thread - 1) / thread, thread>>>(d_bucket, d_bucket2, range);
    cudaDeviceSynchronize();

    // 結果を集める
    gather_kernel<<<(n + thread - 1) / thread, thread>>>(d_key, d_bucket, range, n);
    cudaDeviceSynchronize();

    // 結果の転送
    for (int i=0; i < n; ++i) {
        key[i] = d_key[i];
    }

    // デバイスメモリの解放
    cudaFree(d_key);
    cudaFree(d_bucket);
    cudaFree(d_bucket2);
}

int main() {
    int n = 50;
    int range = 5;
    std::vector<int> key(n);

    // データ生成
    for (int i = 0; i < n; i++) {
        key[i] = rand() % range;
        printf("%d ", key[i]);
    }
    printf("\n");

    // バケットソートの実行
    bucket_sort_gpu(key, range, n);

    for (int i = 0; i < n; i++) {
        printf("%d ", key[i]);
    }
    printf("\n");

    return 0;
}
