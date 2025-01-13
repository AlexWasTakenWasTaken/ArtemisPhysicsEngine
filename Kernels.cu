#include "Kernels.cuh"
#include <cuda_runtime.h>
#include "Vector2D.h"
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void kernelUpdateParticles(Particle* particles, int count, float gravity, float dt, int width, int height) {


    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        particles[idx].velocity.y += gravity * dt;
        particles[idx].position.x += particles[idx].velocity.x * dt;
        particles[idx].position.y -= particles[idx].velocity.y * dt;


        float radius = particles[idx].radius;

        // Left and right walls
        if (particles[idx].position.x - radius < 0.0f) {
            particles[idx].position.x = radius;
            particles[idx].velocity.x = -(0.5) * particles[idx].velocity.x;
        }
        else if (particles[idx].position.x + radius > width) {
            particles[idx].position.x = width - radius;
            particles[idx].velocity.x = -(0.5) * particles[idx].velocity.x;
        }

        // Top and bottom walls
        if (particles[idx].position.y - radius < 0.0f) {
            particles[idx].position.y = radius;
            particles[idx].velocity.y = -(0.5) * particles[idx].velocity.y;
        }
        else if (particles[idx].position.y + radius > height) {
            particles[idx].position.y = height - radius;
            particles[idx].velocity.y = -(0.5) * particles[idx].velocity.y;
        }
    }
}

void gpuUpdateParticles(Particle* d_particles, int count, float gravity, float dt, int width, int height) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelUpdateParticles << <gridSize, BLOCK_SIZE >> > (d_particles, count, gravity, dt, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (Update): %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__
void kernelCollisionDetect(Particle* particles, int count, float dt, float correctionPercent)
{
	const float SLEEP_VELOCITY_THRESHOLD = 0.01f;
	const float SLEEP_TIMEOUT = 5.5f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Particle& p1 = particles[idx];


    for (int j = 0; j < count; ++j) {
        if (j == idx) continue;

        Particle& p2 = particles[j];

        // Calculate distance & overlap
        Vector2D delta = p2.position - p1.position;
        float dist = delta.length();
        float minDist = p1.radius + p2.radius;

        if (dist <= 0.0f || dist >= minDist) {
            continue;
        }

        Vector2D normal = delta / dist; // normalized direction from p1 to p2

        Vector2D rv = p2.velocity - p1.velocity;
        float relVelDot = rv.dot(normal);

        if (relVelDot > 0.0f) {
            continue;
        }

        float e = fminf(p1.restitution, p2.restitution);
        float invM1 = (p1.mass == 0.0f) ? 0.0f : 1.0f / p1.mass;
        float invM2 = (p2.mass == 0.0f) ? 0.0f : 1.0f / p2.mass;

        float z = -(1.0f + e) * relVelDot / (invM1 + invM2);
        Vector2D impulse = normal * z;

        if (!p1.sleeping && invM1 > 0.0f) {
            p1.velocity -= impulse * invM1;
        }
        if (!p2.sleeping && invM2 > 0.0f) {
            p2.velocity += impulse * invM2;
        }

        float penetration = minDist - dist; // how far they overlapped
        if (penetration > 0.0f) {

            Vector2D correction = normal * (correctionPercent * penetration / (invM1 + invM2));

            if (!p1.sleeping && invM1 > 0.0f) {
                p1.position -= correction * invM1;
            }
            if (!p2.sleeping && invM2 > 0.0f) {
                p2.position += correction * invM2;
            }
        }

        float wakeThreshold = 0.1f;
        if (fabsf(j) > wakeThreshold) {
            p1.sleeping = false;
            p1.sleepTimer = 0.0f;
            p2.sleeping = false;
            p2.sleepTimer = 0.0f;
        }
    }

    float speedSq = p1.velocity.dot(p1.velocity);
    if (speedSq < (SLEEP_VELOCITY_THRESHOLD * SLEEP_VELOCITY_THRESHOLD)) {
        p1.sleepTimer += dt;
        if (p1.sleepTimer > SLEEP_TIMEOUT) {
            p1.sleeping = true;
            p1.velocity = Vector2D(0.0f, 0.0f);
        }
    }
    else {
        p1.sleepTimer = 0.0f;
        p1.sleeping = false; // if it moved significantly, it’s awake
    }
}


void gpuCollisionDetect(Particle* d_particles, int count, float dt) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int collisionIterations = 5;
    float correctionPercent = 0.8f;

    for (int i = 0; i < collisionIterations; i++) {
        kernelCollisionDetect << <gridSize, BLOCK_SIZE >> > (d_particles, count, dt, correctionPercent);
        cudaDeviceSynchronize(); // or check errors, etc.
    }
}
