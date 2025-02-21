#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <numeric>

using std::vector;

namespace py = pybind11;


template<typename T> struct matrix {
    size_t n, m;
    vector<vector<T>> a;

    matrix(size_t n_, size_t m_, int val = 0) : n(n_), m(m_), a(n_, vector<T>(m_, val)) {}

    matrix(const T* x, size_t n_, size_t m_) {
        n = n_, m = m_;
        a.resize(n, vector<T>(m));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                a[i][j] = *(x + i * m + j);
            }
        }
    }

    vector<T>& operator[](int k) { return this->a[k]; }

    matrix operator - (matrix& k) { return matrix(*this) -= k; }

    matrix operator * (matrix& k) { return matrix(*this) *= k; }

    matrix& operator -=(matrix& mat) {
        assert(n == mat.n);
        assert(m == mat.m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                a[i][j] -= mat[i][j];
            }
        }
        return *this;
    }

    matrix& operator *= (matrix& mat) {
        assert(m == mat.n);
        int x = n, y = mat.m, z = m;
        matrix<T> c(x, y);
        for (int i = 0; i < x; i++) {
            for (int k = 0; k < z; k++) {
                T r = a[i][k];
                for (int j = 0; j < y; j++) {
                    c[i][j] += mat[k][j] * r;
                }
            }
        }
        return *this = c;
    }

    matrix<T> transpose() {
        matrix<T> result(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = a[j][i];
            }
        }
        return result;
    }

    matrix<float> softmax() {
        matrix<float> result(*this);
        for (int i = 0; i < n; i++) {
            auto& v = result.a[i];
            auto mx = *std::max_element(v.begin(), v.end());
            for (auto& j : v) {
                j -= mx;
                j = std::exp(j);
            }
            auto exp_sum = std::accumulate(v.begin(), v.end(), (float)0.0);
            for (auto& j : v) {
                j /= exp_sum;
            }
        }
        return result;
    }
};

float ce_loss(matrix<float>& Z, matrix<uint8_t>& I_y) {
    int batch = Z.n;
    float sum = 0;
    for (int i = 0; i < batch; i++) {
        float v = Z[i][I_y[i][0]];
        v = -std::log(v);
        sum += v;
    }
    return sum / static_cast<float>(batch);
}

void softmax_regression_epoch_cpp(const float* X, const unsigned char* y,
                                   float* theta, size_t m, size_t n, size_t k,
                                   float lr, size_t batch)
{
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
    // X (m, n); y (m); theta (n, k);
    float alpha = lr / static_cast<float>(batch);
//    float loss = 0;
    for (int e = 0; e < m / batch; e++) {
        matrix<float> batch_X(X + e * batch * n, batch, n);
        matrix<unsigned char> batch_y(y + e * batch, batch, 1);
        matrix<float> t(theta, n, k);

        auto&& XW = batch_X * t;
        auto&& XT = batch_X.transpose();
        auto&& Z = XW.softmax();
        matrix<float> I_y(batch, k);
//        auto loss_i = ce_loss(Z, batch_y);
//        loss += loss_i;

        // X^T @ (Z - I_y)
        for (int i = 0; i < batch; i++) {
            I_y[i][batch_y[i][0]] = 1;
        }
        Z -= I_y;
        auto&& res = XT * Z;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                *(theta + i * k + j) -= res[i][j] * alpha;
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
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
