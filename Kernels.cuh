#pragma once

#include "Particle.h"

static const int BLOCK_SIZE = 256;

void gpuUpdateParticles(Particle* d_particles, int count, float gravity, float dt, int width, int height);
void gpuCollisionDetect(Particle* d_particles, int count, float dt);
