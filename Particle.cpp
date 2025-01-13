#include "Particle.h"
#include <cmath>

Particle::Particle()
    : position(), velocity(), radius(5.0f), mass(1.0f), restitution(0.8f), pressure(0.0f) {}

Particle::Particle(const Vector2D& pos, const Vector2D& vel, float r, float m, float rest, unsigned int id)
    : position(pos), velocity(vel), radius(r), mass(m), restitution(rest), pressure(0.0f), id(id) {}

void Particle::updatePressure(float localDensity, float velocityMagnitude) {
    pressure = localDensity * velocityMagnitude * 0.1f;
}
