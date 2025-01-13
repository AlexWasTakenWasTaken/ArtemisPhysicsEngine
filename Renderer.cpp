#include "Renderer.h"
#include "Particle.h"

#include <iostream>
#include <vector>
#include <unordered_map>


Renderer::Renderer(HINSTANCE hInstance, int width, int height, const char* title)
    : m_hInstance(hInstance), m_isRunning(true) {
    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "ParticleSimulatorWindowClass";

    RegisterClass(&wc);

    m_hwnd = CreateWindowEx(
        0, "ParticleSimulatorWindowClass", title,
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, width, height,
        nullptr, nullptr, hInstance, this);

    if (!m_hwnd) {
        std::cerr << "Failed to create window." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Initialize Direct2D factory
    HRESULT hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &m_pD2DFactory);
    if (FAILED(hr)) {
        std::cerr << "Failed to create Direct2D factory." << std::endl;
        exit(EXIT_FAILURE);
    }
}

Renderer::~Renderer() {
    discardGraphicsResources();
    if (m_pD2DFactory) {
        m_pD2DFactory->Release();
        m_pD2DFactory = nullptr;
    }
    DestroyWindow(m_hwnd);
}

void Renderer::show() {
    ShowWindow(m_hwnd, SW_SHOWDEFAULT);
}

bool Renderer::isRunning() const {
    return m_isRunning;
}

void Renderer::handleMessages() {
    MSG msg = {};
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            m_isRunning = false;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

HWND Renderer::getHWND() const {
    return m_hwnd;
}

HRESULT Renderer::createGraphicsResources() {
    HRESULT hr = S_OK;
    if (!m_pRenderTarget) {
        RECT rc;
        GetClientRect(m_hwnd, &rc);

        D2D1_SIZE_U size = D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top);

        // Create render target
        hr = m_pD2DFactory->CreateHwndRenderTarget(
            D2D1::RenderTargetProperties(),
            D2D1::HwndRenderTargetProperties(m_hwnd, size),
            &m_pRenderTarget
        );

        // Create a black brush
        if (SUCCEEDED(hr)) {
            hr = m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Black), &m_pBlackBrush);
        }
    }
    return hr;
}

void Renderer::discardGraphicsResources() {
    if (m_pRenderTarget) {
        m_pRenderTarget->Release();
        m_pRenderTarget = nullptr;
    }
    if (m_pBlackBrush) {
        m_pBlackBrush->Release();
        m_pBlackBrush = nullptr;
    }
}

void Renderer::draw(ParticleSystem& particleSystem) {
    HRESULT hr = createGraphicsResources();
    if (SUCCEEDED(hr)) {
        m_pRenderTarget->BeginDraw();
        m_pRenderTarget->Clear(D2D1::ColorF(D2D1::ColorF::White));

        for (const auto& particle : particleSystem.getParticles()) {
            float pressureColorScale = particle.pressure * 0.1f;  // Scale factor for color
            D2D1_COLOR_F color = D2D1::ColorF(D2D1::ColorF::Blue);
            if (pressureColorScale > 1.0f) color = D2D1::ColorF(D2D1::ColorF::Red);

            ID2D1SolidColorBrush* brush = nullptr;
            m_pRenderTarget->CreateSolidColorBrush(color, &brush);

            D2D1_ELLIPSE circle = D2D1::Ellipse(D2D1::Point2F(particle.position.x, particle.position.y), particle.radius, particle.radius);
            m_pRenderTarget->FillEllipse(circle, brush);
            brush->Release();
        }

        hr = m_pRenderTarget->EndDraw();
        if (hr == D2DERR_RECREATE_TARGET) {
            discardGraphicsResources();
        }
    }
}

void Renderer::render(ParticleSystem& particleSystem) {
    draw(particleSystem);
}

LRESULT CALLBACK Renderer::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}
