#pragma once
#ifndef PHYSICSENGINE_H
#define PHYSICSENGINE_H

#include "ParticleSystem.h"

class PhysicsEngine {
private:
    float gravitationalAcceleration;  // Gravity constant
    float timeStep;                    // Time step for integration
	int width, height;                 // Width and height of the simulation space

public:
    PhysicsEngine(float gravity, float dt, int width, int height);
    void update(ParticleSystem& system);
};

#endif // PHYSICSENGINE_H
