#include "ParticleSystem.h"
#include <iostream>

ParticleSystem::ParticleSystem() : width(0), height(0){}
ParticleSystem::ParticleSystem(int w, int h) : width(w), height(h) {}

void ParticleSystem::addParticle(const Vector2D& pos, const Vector2D& vel, float radius, float mass, float restitution) {
    Particle newParticle(pos, vel, radius, mass, restitution, numParticle++);
    particles.push_back(newParticle);
}

void ParticleSystem::buildSpatialGrid() {
    spatialGrid.clear();
    for (Particle particle : particles) {
        int cellX = static_cast<int>(particle.position.x / gridSize);
        int cellY = static_cast<int>(particle.position.y / gridSize);
        int cellID = cellX + (cellY * 1000); //unique cell ID
        spatialGrid[cellID].push_back(&particle);
    }
}

void ParticleSystem::resolveCollisions() {
    for (const auto& cell : spatialGrid) {
        const auto& cellParticles = cell.second;
        for (size_t i = 0; i < cellParticles.size(); ++i) {
            for (size_t j = i + 1; j < cellParticles.size(); ++j) {
                Particle& p1 = *cellParticles[i];
                Particle& p2 = *cellParticles[j];
                Vector2D delta = p2.position - p1.position;
                float dist = delta.length();
                float minDist = p1.radius + p2.radius;
                if (dist < minDist) {
                    Vector2D normal = delta.normalized();
                    float relativeVelocity = (p2.velocity - p1.velocity).dot(normal);
                    if (relativeVelocity > 0) continue;  // Particles are moving apart
                    float e = std::min(p1.restitution, p2.restitution);
                    float j = -(1 + e) * relativeVelocity / (1 / p1.mass + 1 / p2.mass);
                    Vector2D impulse = normal * j;
                    p1.velocity -= impulse / p1.mass;
                    p2.velocity += impulse / p2.mass;
                }
            }
        }
    }
}

void ParticleSystem::update() {
    buildSpatialGrid();
    resolveCollisions();
    for (auto& particle : particles) {
        particle.position += particle.velocity;  // Basic position update
    }
}

std::vector<Particle>& ParticleSystem::getParticles() {
	return particles;
}