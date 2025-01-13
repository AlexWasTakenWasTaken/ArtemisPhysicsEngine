#pragma once

#ifndef RENDERER_H
#define RENDERER_H

#include <windows.h>
#include <d2d1.h>
#include "ParticleSystem.h"

#pragma comment(lib, "d2d1")

class Renderer {
public:
    Renderer(HINSTANCE hInstance, int width, int height, const char* title);
    ~Renderer();

    bool isRunning() const;
    void handleMessages();
    HWND getHWND() const;
    void show();
    void render(ParticleSystem& particleSystem);

private:
    HINSTANCE m_hInstance;
    HWND m_hwnd;
    bool m_isRunning;

    // Direct2D resources
    ID2D1Factory* m_pD2DFactory = nullptr;
    ID2D1HwndRenderTarget* m_pRenderTarget = nullptr;
    ID2D1SolidColorBrush* m_pBlackBrush = nullptr;

    HRESULT createGraphicsResources();
    void discardGraphicsResources();
    void draw(ParticleSystem& particleSystem);

    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
};

#endif  // RENDERER_H
