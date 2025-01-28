
#include <thrust/device_vector.hpp>

/// @brief
template <typename T, int N>
__global__ void mass_action(const T* qpts, const T* qwts, const T* c0_in, T* f2)
{

  // TODO: allow Q to vary independently of N
  constexpr int Q = N + 1;

  __shared__ T dofs[N + 1][N + 1];
  __shared__ T f0[Q][N + 1];
  __shared__ T qvals[Q][Q];
  __shared__ T f1[N + 1][Q];
  // Output: f2[N + 1][N + 1]

  {
    // DOFs
    int a1 = threadIdx.x;
    int a2 = threadIdx.y;

    // Copy input dofs to shared memory
    dofs[a1][a2] = c0_in[a1 * (N + 1) + a2];

    __syncthreads();
  }

  {
    // QPs
    int i1 = threadIdx.x;

    // DOFs
    int a2 = threadIdx.y;

    T x = qpts[i1];
    T r = (1 - x) / x;
    T b = 1.0;
    for (int i = 0; i < N; ++i)
      b *= x;
    for (int a1 = 0; a1 < (N + 1); ++a1)
    {
      f0[i1][a2] += b * dofs[a1][a2];
      b *= r * (N - a1) / (a1 + 1);
    }
  }
  __syncthreads();
  {
    // QPs
    int i1 = threadIdx.x;
    int i2 = threadIdx.y;

    T x = qpts[i2];
    T r = (1 - x) / x;
    T b = 1.0; // b = x^n
    for (int i = 0; i < N; ++i)
      b *= x;
    for (int a2 = 0; a2 < (N + 1); ++a2)
    {
      qvals[i1][i2] += w * f0[i1][a2];
      b *= r * (N - a2) / (1 + a2);
    }
  }
  __syncthreads();

  // qvals contains values at quadrature points
  // TODO: apply geometry here

  {
    // DOFs
    int a1 = threadIdx.x;
    // QPs
    int i2 = threadIdx.y;

    for (int i1 = 0; i1 < Q; ++i1)
    {
      T x = qpts[i1];
      T w = qwts[i1];
      for (int i = 0; i < N - a1; ++i)
        w *= x;
      for (int i = 0; i < a1; ++i)
        w *= (1 - x) * (N - i) / (1 + i);
      f1[a1][i2] += w * qvals[i1][i2];
    }
  }
  __syncthreads();
  {
    // DOFs
    int a1 = threadIdx.x;
    int a2 = threadIdx.y;

    for (int i2 = 0; i2 < Q; ++i2)
    {
      T x = qpts[i2];
      T w = qwts[i2];
      for (int i = 0; i < N - a2; ++i)
        w *= x;
      for (int i = 0; i < a2; ++i)
        w *= (1 - x) * (N - i) / (1 + i);
      f2[a1 * (N + 1) + a2] += w * f1[a1][i2];
    }
  }
}

using T = double;

int main()
{
  int n = 4;

  // Create device vectors for qpts and qwts
  thrust::device_vector<T> qpts0
      = {0.04691007703066802, 0.23076534494715845, 0.5, 0.7692346550528415,
         0.9530899229693319};
  thrust::device_vector<T> qpts1
      = {0.03980985705146878, 0.1980134178736081, 0.4379748102473862,
         0.695464273353636, 0.9014649142011736};
  thrust::device_vector<T> qwts0
      = {0.11846344252809478, 0.2393143352496831, 0.2844444444444443,
         0.2393143352496831, 0.11846344252809478};
  thrust::device_vector<T> qwts1
      = {0.09678159022665209, 0.16717463809436933, 0.14638698708466968,
         0.07390887007261666, 0.01574791452169229};

  // Create input vector
  thrust::device_vector<T> dofs_in(n * n), dofs_out(n * n);

  assert(n == 4);
  mass_action<T, 4><<<1, n>>>(qpts0, qwts0, qpts1, qwts1, dofs_in, dofs_out);
}
