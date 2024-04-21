#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0);
#pragma omp parallel for
  for (int i=0; i<n; i++){
#pragma omp atomic
    bucket[key[i]]++;
  }

  std::vector<int> offset(range,0);
#pragma omp parallel for
  for (int i=1; i<range; i++){
#pragma omp critical
    offset[i] = offset[i-1] + bucket[i-1];
  }

#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (int b=bucket[i]; b>0; b--) {
#pragma omp critical
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
