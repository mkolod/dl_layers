#include <array>
#import <cmath>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>
#include <omp.h>



template <typename T>
class Tensor4D {
public:

    // constructor
    Tensor4D(const size_t N,
             const size_t C,
             const size_t H,
             const size_t W) :
             ptr(new T[N*C*H*W]),
             N(N), C(C), H(H), W(W) {
    }

    Tensor4D() {

    }

    ~Tensor4D() {
        safe_delete(ptr);
    }

    // copy constructor
    Tensor4D(const Tensor4D& other) = default;

    // copy assignment operator
    Tensor4D& operator=(const Tensor4D& other) = default;

    // move constructor
    Tensor4D(Tensor4D&& other) :
    N(other.N), C(other.C), H(other.H), W(other.W) {
       ptr = other.ptr;
       other.ptr = nullptr;
    }

    // move assignment operator
    Tensor4D& operator=(Tensor4D&& other) {

        N = other.N;
        C = other.C;
        H = other.H;
        W = other.W;

        if (this != &other) {
            safe_delete(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    T at(const size_t n, const size_t c, const size_t h, const size_t w) const {
        return ptr[calcOffset(n, c, h, w)];
    }

    T& at(const size_t n, const size_t c, const size_t h, const size_t w) {
        return ptr[calcOffset(n, c, h, w)];
    }

    std::array<size_t, 4> dims() const {
        return {{N, C, H, W}};
    }

    T* data() {
        return ptr;
    }

private:

    size_t calcOffset(const size_t n, const size_t c, const size_t h, const size_t w) const {
        return n*C*H*W + c*H*W + h*W + w;
    }

    void safe_delete(T* ptr) {
        if (ptr != nullptr) {
            delete [] ptr;
        }
    }

    T* ptr;
    const size_t N;
    const size_t C;
    const size_t H;
    const size_t W;
};

template <typename T>
std::tuple<Tensor4D<T>, std::vector<T>, std::vector<T>>
batchNorm(
        const Tensor4D<T>& data,
        const std::vector<T>& running_mean,
        const std::vector<T>& running_std,
        T scale = 1.0,
        T shift = 0.0,
        T momentum = 0.1,
        T eps = 1e-5,
        bool inference = true) {

   // Have to unpack manually rather than with "auto [N, C, H, W] = data.dims()"
   // because the latter seems to make OpenMP unhappy.
   const auto dims = data.dims();
   const size_t N = dims[0];
   const size_t C = dims[1];
   const size_t H = dims[2];
   const size_t W = dims[3];

   std::vector<T> sample_mean(running_mean.size());
   std::vector<T> sample_std(running_std.size());

   const size_t nhw = N*H*W;

   // parallelize over channels
   #pragma omp parallel for
   for (size_t c = 0; c < C; ++c) {

       size_t count = 0;
       T m;
       T s = 0;

       // Welford's formula for stable mean and std
       for (size_t n = 0; n < N; ++n) {
           for (size_t h = 0; h < H; ++h) {
               for (size_t w = 0; w < W; ++w) {
                   count++;
                   T datum = data.at(n, c, h, w);
                   if (count == 1) {
                       m = datum;
                   } else {
                       T mNext = m + (datum - m) / count;
                       s = s + (datum - m)*(datum - mNext);
                       m = mNext;
                   }

               }
           }
       }
       sample_mean.at(c) = m;
       sample_std.at(c) = sqrt(s / (count - 1));
   }

   Tensor4D<T> output(N, C, H, W);

    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    if (inference) {
                        output.at(n, c, h, w) =
                                scale * (data.at(n, c, h, w) - running_mean.at(c)) / (running_std.at(c) + eps) + shift;
                    } else {
                        output.at(n, c, h, w) =
                                scale * (data.at(n, c, h, w) - sample_mean.at(c)) / (sample_std.at(c) + eps) + shift;
                    }
                }
            }
        }
    }

    std::vector<T> new_running_mean(running_mean.size());
    std::vector<T> new_running_std(running_std.size());
    for (size_t idx = 0; idx < running_mean.size(); ++idx) {
        if (!inference) {
            new_running_mean.at(idx) = running_mean.at(idx) * momentum + (1 - momentum) * sample_mean.at(idx);
            new_running_std.at(idx) = running_std.at(idx) * momentum + (1 - momentum) * sample_std.at(idx);
        }
    }

    return std::make_tuple(std::move(output), new_running_mean, new_running_std);
}


int main() {

    const std::vector<float> running_mean { 1.41f, 3.14f, 2.72f};
    const std::vector<float> running_std { 1.0f, 2.0f, 3.0f };
    Tensor4D<float> input(128, 3, 224, 224);
    auto [normed_input, new_running_mean, new_running_std] = batchNorm<float>(input, running_mean, running_std);
    std::cout << normed_input.data()[0] << std::endl;
    return 0;
}
