#pragma once

#ifndef VECTOR2D_H
#define VECTOR2D_H

#include <cuda_runtime.h>
#include <cmath>

class Vector2D {
public:
    float x;
    float y;

    __host__ __device__ Vector2D() : x(0.0f), y(0.0f) {}
    __host__ __device__ Vector2D(float xVal, float yVal) : x(xVal), y(yVal) {}

    __host__ __device__ Vector2D operator+(const Vector2D& other) const {
        return Vector2D(x + other.x, y + other.y);
    }

    __host__ __device__ Vector2D& operator+=(const Vector2D& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    __host__ __device__ Vector2D operator-(const Vector2D& other) const {
        return Vector2D(x - other.x, y - other.y);
    }

    __host__ __device__ Vector2D& operator-=(const Vector2D& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    __host__ __device__ Vector2D operator*(float scalar) const {
        return Vector2D(x * scalar, y * scalar);
    }

    __host__ __device__ Vector2D operator/(float scalar) const {
        return Vector2D(x / scalar, y / scalar);
    }

    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y);
    }

    __host__ __device__ Vector2D normalized() const {
        float len = length();
        return (len > 0) ? Vector2D(x / len, y / len) : Vector2D(0.0f, 0.0f);
    }

    __host__ __device__ float dot(const Vector2D& other) const {
        return x * other.x + y * other.y;
    }

    __host__ __device__ void reset() {
        x = 0.0f;
        y = 0.0f;
    }
};

#endif  // VECTOR2D_H
