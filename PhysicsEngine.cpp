#include "PhysicsEngine.h"
#include "Kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

PhysicsEngine::PhysicsEngine(float gravity, float dt, int width, int height)
    : gravitationalAcceleration(gravity * 1/dt), timeStep(dt), width(width), height(height) {}

void PhysicsEngine::update(ParticleSystem& system) {
    auto& particles = system.getParticles();
    int count = static_cast<int>(particles.size());

    if (count == 0) return;

    Particle* d_particles = nullptr;

    // allocate GPU mem
    cudaError_t allocStatus = cudaMalloc(&d_particles, count * sizeof(Particle));
    if (allocStatus != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(allocStatus) << std::endl;
        return;
    }

    cudaMemcpy(d_particles, particles.data(), count * sizeof(Particle), cudaMemcpyHostToDevice);

    gpuUpdateParticles(d_particles, count, gravitationalAcceleration, timeStep, width, height);

    gpuCollisionDetect(d_particles, count, timeStep);

    cudaMemcpy(particles.data(), d_particles, count * sizeof(Particle), cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
}
