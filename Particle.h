#pragma once

#ifndef PARTICLE_H
#define PARTICLE_H

#include "Vector2D.h"    
class Particle {
public:
    Vector2D position;
    Vector2D velocity; 
    float radius;           
    float mass;             
    float restitution;

	unsigned int id; 

	bool sleeping = false;
	float sleepTimer = 0.0f;

    float pressure;

    Particle();
    Particle(const Vector2D& pos, const Vector2D& vel, float r, float m, float rest, unsigned int id);

    void updatePressure(float localDensity, float velocityMagnitude);
};

#endif // PARTICLE_H


