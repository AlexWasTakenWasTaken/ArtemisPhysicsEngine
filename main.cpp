#include "ParticleSystem.h"
#include "PhysicsEngine.h"
#include "Renderer.h"
#include <chrono>
#include <thread>

const float dt = 0.02f;  // Time step per frame
const int height = 1000;
const int width = 1000;


int main(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    Renderer renderer(hInstance, width, height, "Particle Simulator");

    ParticleSystem particleSystem(width, height);

    PhysicsEngine physicsEngine(0.0981f, dt, width, height);


    renderer.show();

    while (renderer.isRunning()) {
        auto startTime = std::chrono::high_resolution_clock::now();

        particleSystem.addParticle(Vector2D(100.0f, 800.0f), Vector2D(100.0f, 0.0f), 3);
        particleSystem.addParticle(Vector2D(900.0f, 800.0f), Vector2D(-100.0f, 0.0f), 3);
        physicsEngine.update(particleSystem);
        

        renderer.render(particleSystem);
        renderer.handleMessages();

        // frame limitign
        auto endTime = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(dt * 1000)) -
            (endTime - startTime));
    }

    return 0;
}
