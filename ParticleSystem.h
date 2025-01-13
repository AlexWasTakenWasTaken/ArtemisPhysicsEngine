#pragma once
#include <vector>
#include <unordered_map>
#include "Particle.h"

class ParticleSystem {
private:
    std::vector<Particle> particles;  // Container for particle objects
    float gridSize = 10.0f;           // Grid cell size for spatial partitioning
    std::unordered_map<int, std::vector<Particle*>> spatialGrid; // Spatial partitioning grid

	unsigned int numParticle = 0;
    
    int width = 0;
	int height = 0;

public:
    ParticleSystem();
    ParticleSystem(int w, int h);

    void addParticle(const Vector2D& pos, const Vector2D& vel, float radius = 5.0f, float mass = 1.0f, float restitution = 0.5f);
    void update();
    void buildSpatialGrid();
    void resolveCollisions();
    std::vector<Particle>& getParticles();
    std::unordered_map<int, std::vector<Particle*>>& getSpatialGrid();
    size_t size() const { return particles.size(); }
};
