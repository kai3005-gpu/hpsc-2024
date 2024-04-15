#include <cstdio>

int main() {
  int n = 10000;
  double dx = 1. / n;
  double pi = 0;
  #pragma omp parallel for
  for (int i=0; i<n; i++) {
    double x = (i + 0.5) * dx;
    pi += 4.0 / (1.0 + x * x) * dx;
  }
  printf("%17.15f\n",pi);
}
