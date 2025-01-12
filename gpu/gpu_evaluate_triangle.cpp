
#include <thrust/device_vector.h>

/// @brief Mass operator using Bernstein Polynomials
template <typename T, int N>
__global__ void mass_action(const T* qpts0, const T* qwts0, const T* qpts1, const T* qwts1,
                            const T* c0_in, T* f2)
{
  // DOFs
  int a1 = threadIdx.x;
  int a2 = threadIdx.y;

  // TODO: allow Q to vary independently of N
  constexpr int Q = N + 1;

  __shared__ T c0[N + 1][N + 1];
  __shared__ T c1[N + 1][Q];
  __shared__ T c2[Q][Q];
  __shared__ T f1[N + 1][Q];
  // f2[N + 1][N + 1]

  // Copy input dofs to shared memory
  c0[a1][a2] = c0_in[a1 * (N + 1) + a2];

  // i2 -> QP
  int i2 = threadIdx.y;
  T p = qpts0[i2];
  T s = 1 - p;
  T w = 1.0; // w = s^(n - a1)
  for (int i = 0; i < (N - a1); ++i)
    w *= s;

  // c1 = evalstep(c0, l=2, q)
  T r = p / s;
  T v = 0.0;
  for (int a2 = 0; a2 < (N + 1 - a1); ++a2)
  {
    v += w * c0[a1][a2];
    w *= r * (N - a1 - a2) / (1 + a2);
  }
  c1[a1][i2] = v;

  __syncthreads();

  // i1 -> QP
  int i1 = threadIdx.x;
  p = qpts1[i1];
  s = 1 - p;
  w = 1.0; // w = s^n
  for (int i = 0; i < N; ++i)
    w *= s;
  r = p / s;
  v = 0.0;
  for (int a1 = 0; a1 < (N + 1); ++a1)
  {
    v += w * c1[a1][i2];
    w *= r * (N - a1) / (1 + a1);
  }
  c2[i1][i2] = v;

  __syncthreads();

  // c2 contains values at quadrature points
  // TODO: apply geometry here

  // a1 -> DOF
  a1 = threadIdx.x;

  v = 0;
  for (int i1 = 0; i1 < Q; ++i1)
  {
    p = qpts1[i1];
    w = qwts1[i1];
    s = 1 - p;
    r = p / s;
    for (int i = 0; i < N; ++i)
      w *= s;
    for (int i = 0; i < a1; ++i)
      w *= r * (N - i) / (1 + i);
    v += w * c2[i1][i2];
  }
  f1[a1][i2] = v;

  printf("f1[%d][%d] = %f\n", a1, i2, f1[a1][i2]);

  __syncthreads();

  // a2 -> dof
  a2 = threadIdx.y;
  v = 0.0;
  if ((a2 + a1) < (N + 1))
  {
    for (int i2 = 0; i2 < Q; ++i2)
    {
      p = qpts0[i2];
      w = qwts0[i2];
      s = 1 - p;
      for (int i = 0; i < (N - a1); ++i)
        w *= s;

      r = p / s;
      for (int i = 0; i < a2; ++i)
        w *= r * (N - a1 - i) / (1 + i);

      v += w * f1[a1][i2];
    }
  }

  printf("f2[%d][%d] = %f\n", a1, a2, v);
  f2[a1 * (N + 1) + a2] = v;
}

using T = double;

int main()
{
  int n = 4;

  // Create device vectors for qpts and qwts
  std::vector<T> qpts0
      = {0.04691007703066802, 0.23076534494715845, 0.5, 0.7692346550528415, 0.9530899229693319};
  thrust::device_vector<T> qpts0_device = qpts0;

  std::vector<T> qpts1 = {0.03980985705146878, 0.1980134178736081, 0.4379748102473862,
                          0.695464273353636, 0.9014649142011736};
  thrust::device_vector<T> qpts1_device = qpts1;

  std::vector<T> qwts0 = {0.11846344252809478, 0.2393143352496831, 0.2844444444444443,
                          0.2393143352496831, 0.11846344252809478};
  thrust::device_vector<T> qwts0_device = qwts0;

  std::vector<T> qwts1 = {0.09678159022665209, 0.16717463809436933, 0.14638698708466968,
                          0.07390887007261666, 0.01574791452169229};
  thrust::device_vector<T> qwts1_device = qwts1;

  // Create input vector
  thrust::device_vector<T> dofs_in((n + 1) * (n + 1)), dofs_out((n + 1) * (n + 1));
  thrust::fill(dofs_in.begin(), dofs_in.end(), 0.0);
  dofs_in[0] = 1.0;

  assert(n == 4);
  dim3 blocksize(n + 1, n + 1);
  mass_action<T, 4><<<1, blocksize>>>(
      thrust::raw_pointer_cast(qpts0_device.data()), thrust::raw_pointer_cast(qwts0_device.data()),
      thrust::raw_pointer_cast(qpts1_device.data()), thrust::raw_pointer_cast(qwts1_device.data()),
      thrust::raw_pointer_cast(dofs_in.data()), thrust::raw_pointer_cast(dofs_out.data()));

  std::vector<T> output((n + 1) * (n + 1));
  thrust::copy(dofs_out.begin(), dofs_out.end(), output.begin());
  for (auto q : output)
    std::cout << q << "\n";
}
