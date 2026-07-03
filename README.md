# Gesture Controlled Game — Origo 2025

A neon-themed gesture-controlled arcade game built for **Origo 2025** (NIT Calicut's tech fest). Uses **MediaPipe Hands** for real-time hand tracking and **Pygame** for rendering.

## How It Works

The webcam captures hand gestures — the game detects **hand position** (X-axis for movement), **pinch gesture** for actions, and **open palm** for special moves. A neon visual aesthetic with particle effects creates an immersive arcade experience.

## Features

- Real-time hand tracking via MediaPipe (running in a separate thread)
- Pinch detection for primary interactions
- Open palm gesture recognition
- Neon/cyberpunk visual style with particle systems
- Pygame-based rendering

## Requirements

- Python 3.8+
- Webcam
- OpenCV
- MediaPipe
- Pygame
- NumPy

## Setup

```bash
git clone https://github.com/bip-krishna/Gesture_controlled_game_origo_2025.git
cd Gesture_controlled_game_origo_2025
pip install opencv-python mediapipe pygame numpy
python improved_neon_gesture_game.py
```

## Controls

| Gesture | Action |
|---|---|
| Move hand left/right | Horizontal movement |
| Pinch (thumb + index) | Primary action / select |
| Open palm | Special action / pause |
