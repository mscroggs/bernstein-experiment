
#include <iomanip>
#include <iostream>
#include <thrust/device_vector.h>

/// @brief Laplacian operator using Bernstein Polynomials
/// @param qdata - Quadrature points and weights for x and y - size 4*(N + 1)
/// @param c0_in - Input dofs, arranged in a 2D array (upper triangle)
/// @param f2 - Output dofs, arranged same as input
template <typename T, int M>
__global__ void laplacian_action(const T* qdata, const T* c0_in, T* f2)
{
  // a1, a2 -> DOFs
  int a1 = threadIdx.x;
  int a2 = threadIdx.y;

  // TODO: allow Q to vary independently of N
  constexpr int N = M - 1;
  constexpr int Q = N + 1;

  __shared__ T c0[M + 1][M + 1];
  __shared__ T c1x[N + 1][Q];
  __shared__ T c1y[N + 1][Q];
  __shared__ T c2x[Q][Q];
  __shared__ T c2y[Q][Q];
  __shared__ T f1x[N + 1][Q];
  __shared__ T f1y[N + 1][Q];
  // f2[M + 1][M + 1]

  // Copy input dofs to shared memory
  c0[a1][a2] = c0_in[a1 * (M + 1) + a2];

  // Copy tables to shared memory
  __shared__ T qpts0[Q], qwts0[Q], qpts1[Q], qwts1[Q];
  if (a1 == 0)
    qpts0[a2] = qdata[a2];
  else if (a1 == 1)
    qwts0[a2] = qdata[Q + a2];
  else if (a1 == 2)
    qpts1[a2] = qdata[2 * Q + a2];
  else if (a1 == 3)
    qwts1[a2] = qdata[3 * Q + a2];

  // i2 -> QP
  int i2 = threadIdx.y;
  {
    T p = qpts0[i2];
    T s = 1 - p;
    T w = 1.0; // w = s^(n - a1)
    for (int i = 0; i < (N - a1); ++i)
      w *= s;

    // c1 = evalstep(c0, l=2, q)
    T r = p / s;
    T vc1x = 0.0;
    T vc1y = 0.0;
    for (int a2 = 0; a2 < (N + 1 - a1); ++a2)
    {
      vc1x += w * (c0[a1][a2] - c0[a1 + 1][a2]);
      vc1y += w * (c0[a1][a2] - c0[a1][a2 + 1]);
      w *= r * (N - a1 - a2) / (1 + a2);
    }
    c1x[a1][i2] = vc1x;
    c1y[a1][i2] = vc1y;
    __syncthreads();
  }

  // i1 -> QP
  int i1 = threadIdx.x;
  {
    T p = qpts1[i1];
    T s = 1 - p;
    T w = 1.0; // w = s^n
    for (int i = 0; i < N; ++i)
      w *= s;
    T r = p / s;
    T vc2x = 0.0;
    T vc2y = 0.0;
    for (int a1 = 0; a1 < (N + 1); ++a1)
    {
      vc2x += w * c1x[a1][i2];
      vc2y += w * c1y[a1][i2];
      w *= r * (N - a1) / (1 + a1);
    }
    c2x[i1][i2] = vc2x;
    c2y[i1][i2] = vc2y;
    __syncthreads();
  }

  // c2 contains values at quadrature points
  // TODO: apply geometry here

  // a1 -> DOF
  a1 = threadIdx.x;
  {
    T vf1x = 0;
    T vf1y = 0;
    for (int i1 = 0; i1 < Q; ++i1)
    {
      T p = qpts1[i1];
      T w = qwts1[i1];
      for (int i = 0; i < N - a1; ++i)
        w *= (1 - p);
      for (int i = 0; i < a1; ++i)
        w *= p * (N - i) / (1 + i);
      vf1x += w * c2x[i1][i2];
      vf1y += w * c2y[i1][i2];
    }
    f1x[a1][i2] = vf1x;
    f1y[a1][i2] = vf1y;
    __syncthreads();
  }

  // a2 -> dof
  a2 = threadIdx.y;
  {
    T vf2x = 0;
    T vf2y = 0;
    if ((a2 + a1) < (N + 1))
    {
      for (int i2 = 0; i2 < Q; ++i2)
      {
        T p = qpts0[i2];
        T w = qwts0[i2];
        for (int i = 0; i < (N - a1 - a2); ++i)
          w *= (1 - p);
        for (int i = 0; i < a2; ++i)
          w *= p * (N - a1 - i) / (1 + i);
        vf2x += w * f1x[a1][i2];
        vf2y += w * f1y[a1][i2];
      }
    }
  }
  f2[a1 * (M + 1) + a2] = -(vf2x + vf2y);
  __syncthreads();
  f2[(a1 + 1) * (M + 1) + a2] += vf2x;
  __syncthreads();
  f2[a1 * (M + 1) + a2 + 1] += vf2y;
}

/// @brief Mass operator using Bernstein Polynomials
/// @param qdata - Quadrature points and weights for x and y - size 4*(N + 1)
/// @param c0_in - Input dofs, arranged in a 2D array (upper triangle)
/// @param f2 - Output dofs, arranged same as input
template <typename T, int N>
__global__ void mass_action(const T* qdata, const T* c0_in, T* f2)
{
  // a1, a2 -> DOFs
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

  // Copy tables to shared memory
  __shared__ T qpts0[Q], qwts0[Q], qpts1[Q], qwts1[Q];
  if (a1 == 0)
    qpts0[a2] = qdata[a2];
  else if (a1 == 1)
    qwts0[a2] = qdata[Q + a2];
  else if (a1 == 2)
    qpts1[a2] = qdata[2 * Q + a2];
  else if (a1 == 3)
    qwts1[a2] = qdata[3 * Q + a2];

  // i2 -> QP
  int i2 = threadIdx.y;
  {
    T p = qpts0[i2];
    T s = 1 - p;
    T w = 1.0; // w = s^(n - a1)
    for (int i = 0; i < (N - a1); ++i)
      w *= s;

    // c1 = evalstep(c0, l=2, q)
    T r = p / s;
    T vc1 = 0.0;
    for (int a2 = 0; a2 < (N + 1 - a1); ++a2)
    {
      vc1 += w * c0[a1][a2];
      w *= r * (N - a1 - a2) / (1 + a2);
    }
    c1[a1][i2] = vc1;
    __syncthreads();
  }

  // i1 -> QP
  int i1 = threadIdx.x;
  {
    T p = qpts1[i1];
    T s = 1 - p;
    T w = 1.0; // w = s^n
    for (int i = 0; i < N; ++i)
      w *= s;
    T r = p / s;
    T vc2 = 0.0;
    for (int a1 = 0; a1 < (N + 1); ++a1)
    {
      vc2 += w * c1[a1][i2];
      w *= r * (N - a1) / (1 + a1);
    }
    c2[i1][i2] = vc2;
    __syncthreads();
  }

  // c2 contains values at quadrature points
  // TODO: apply geometry here

  // a1 -> DOF
  a1 = threadIdx.x;
  {
    T vf1 = 0;
    for (int i1 = 0; i1 < Q; ++i1)
    {
      T p = qpts1[i1];
      T w = qwts1[i1];
      for (int i = 0; i < N - a1; ++i)
        w *= (1 - p);
      for (int i = 0; i < a1; ++i)
        w *= p * (N - i) / (1 + i);
      vf1 += w * c2[i1][i2];
    }
    f1[a1][i2] = vf1;
    __syncthreads();
  }

  // a2 -> dof
  a2 = threadIdx.y;
  {
    T vf2 = 0;
    if ((a2 + a1) < (N + 1))
    {
      for (int i2 = 0; i2 < Q; ++i2)
      {
        T p = qpts0[i2];
        T w = qwts0[i2];
        for (int i = 0; i < (N - a1 - a2); ++i)
          w *= (1 - p);
        for (int i = 0; i < a2; ++i)
          w *= p * (N - a1 - i) / (1 + i);
        vf2 += w * f1[a1][i2];
      }
    }
    f2[a1 * (N + 1) + a2] = vf2;
  }
}

using T = double;

int main()
{
  int n = 4;

  thrust::device_vector<T> q_device((n + 1) * 4);

  // Create device vectors for qpts and qwts
  std::vector<T> qpts0
      = {0.04691007703066802, 0.23076534494715845, 0.5, 0.7692346550528415, 0.9530899229693319};
  std::vector<T> qwts0 = {0.11846344252809478, 0.2393143352496831, 0.2844444444444443,
                          0.2393143352496831, 0.11846344252809478};
  std::vector<T> qpts1 = {0.03980985705146878, 0.1980134178736081, 0.4379748102473862,
                          0.695464273353636, 0.9014649142011736};
  std::vector<T> qwts1 = {0.09678159022665209, 0.16717463809436933, 0.14638698708466968,
                          0.07390887007261666, 0.01574791452169229};
  thrust::copy(qpts0.begin(), qpts0.end(), q_device.begin());
  thrust::copy(qwts0.begin(), qwts0.end(), q_device.begin() + (n + 1));
  thrust::copy(qpts1.begin(), qpts1.end(), q_device.begin() + 2 * (n + 1));
  thrust::copy(qwts1.begin(), qwts1.end(), q_device.begin() + 3 * (n + 1));

  // Create input vector
  thrust::device_vector<T> dofs_in((n + 1) * (n + 1)), dofs_out((n + 1) * (n + 1));
  thrust::fill(dofs_in.begin(), dofs_in.end(), 0.0);
  dofs_in[0] = 1.0;

  assert(n == 4);
  dim3 blocksize(n + 1, n + 1);
  mass_action<T, 4><<<1, blocksize>>>(thrust::raw_pointer_cast(q_device.data()),
                                      thrust::raw_pointer_cast(dofs_in.data()),
                                      thrust::raw_pointer_cast(dofs_out.data()));

  std::vector<T> output((n + 1) * (n + 1));
  thrust::copy(dofs_out.begin(), dofs_out.end(), output.begin());

  for (int i = 0; i < (n + 1); ++i)
  {
    std::cout << "[";
    for (int j = 0; j < (n + 1); ++j)
      std::cout << std::setprecision(4) << std::setw(10) << dofs_out[i * (n + 1) + j] << " ";
    std::cout << "]\n";
  }
}
