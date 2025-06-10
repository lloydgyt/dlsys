#include <assert.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string.h>

namespace py = pybind11;

// C(m * k) = A(m * n) @ B(n * k)
// changes C
void matrix_mult(const float *A, const float *B, float *C, size_t m, size_t n,
                 size_t k) {
  // TODO: use blocking to boost
  // NOTE: C is set to 0 at this point
  for (size_t h = 0; h < n; h++) {
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        C[i * k + j] += A[i * n + h] * B[h * k + j];
      }
    }
  }
}

// modify H(m by n) by softmax
void softmax(float *H, size_t m, size_t n) {
  for (size_t i = 0; i < m; i++) {
    float sum = 0;
    for (size_t j = 0; j < n; j++) {
      H[i * n + j] = exp(H[i * n + j]);
      sum += H[i * n + j];
    }
    for (size_t j = 0; j < n; j++) {
      H[i * n + j] /= sum;
    }
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
  /**
   * A C++ version of the softmax regression epoch code.  This should run a
   * single epoch over the data defined by X and y (and sizes m,n,k), and
   * modify theta in place.  Your function will probably want to allocate
   * (and then delete) some helper arrays to store the logits and gradients.
   *
   * Args:
   *     X (const float *): pointer to X data, of size m*n, stored in row
   *          major (C) format
   *     y (const unsigned char *): pointer to y data, of size m
   *     theta (float *): pointer to theta data, of size n*k, stored in row
   *          major (C) format
   *     m (size_t): number of examples
   *     n (size_t): input dimension
   *     k (size_t): number of classes
   *     lr (float): learning rate / SGD step size
   *     batch (int): SGD minibatch size
   *
   * Returns:
   *     (None)
   */

  /// BEGIN YOUR CODE
  float X_theta[batch * k];
  // printf("stack array begins at: %p\n", X_theta);
  for (size_t base = 0; base + batch - 1 < m; base += batch) {
    const float *curr_X = X + base * n;
    const unsigned char *curr_y = y + base;
    // TODO: use stack array or malloc free?
    // get softmax (Z) using batch X
    memset(X_theta, 0, sizeof(X_theta));
    matrix_mult(curr_X, theta, X_theta, batch, n, k);
    softmax(X_theta, batch, k);
    for (size_t i = 0; i < batch; i++) {
      X_theta[i * k + curr_y[i]] -= 1;
    }
    // update using d_theta
    for (size_t h = 0; h < batch; h++) {
      for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < k; j++) {
          theta[i * k + j] -=
              lr * (curr_X[h * n + i] * X_theta[h * k + j]) / ((float)batch);
        }
      }
    }
  }
  /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
  m.def(
      "softmax_regression_epoch_cpp",
      [](py::array_t<float, py::array::c_style> X,
         py::array_t<unsigned char, py::array::c_style> y,
         py::array_t<float, py::array::c_style> theta, float lr, int batch) {
        softmax_regression_epoch_cpp(
            static_cast<const float *>(X.request().ptr),
            static_cast<const unsigned char *>(y.request().ptr),
            static_cast<float *>(theta.request().ptr), X.request().shape[0],
            X.request().shape[1], theta.request().shape[1], lr, batch);
      },
      py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"),
      py::arg("batch"));
}
