import os, sys, math, time, random, threading
import numpy as np
import pygame
import cv2
import mediapipe as mp

# ---------------------------
# Gesture Detector (for hand control)
# ---------------------------
class GestureDetector:
    def __init__(self, cam_index=0, width=640, height=480, process_every_n_frames=2):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.running = True
        self.frame_count = 0
        self.process_every_n_frames = process_every_n_frames

        self.hand_present = False
        self.hand_x_norm = 0.5
        self.pinch_active = False
        self.open_palm = False
        self.last_update_ts = 0.0
        self.last_frame = np.zeros((height, width, 3), dtype=np.uint8)

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            frame = cv2.flip(frame, 1)
            self.last_frame = frame.copy()
            self.frame_count += 1

            if self.frame_count % self.process_every_n_frames != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                cx = (lm[0].x + lm[9].x) / 2.0
                thumb_tip = np.array([lm[4].x, lm[4].y])
                index_tip = np.array([lm[8].x, lm[8].y])
                pinch_dist = np.linalg.norm(thumb_tip - index_tip)
                ref_scale = np.linalg.norm(np.array([lm[0].x, lm[0].y]) - np.array([lm[9].x, lm[9].y])) + 1e-6
                pinch_ratio = pinch_dist / ref_scale

                def finger_extended(tip, pip): return lm[tip].y < lm[pip].y
                extended = 0
                for t, p in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                    if finger_extended(t, p): extended += 1
                thumb_ext = np.linalg.norm(np.array([lm[4].x, lm[4].y]) - np.array([lm[2].x, lm[2].y])) > 0.06
                if thumb_ext: extended += 1

                self.hand_present = True
                self.hand_x_norm = float(np.clip(cx, 0.0, 1.0))
                self.pinch_active = pinch_ratio < 0.55
                self.open_palm = extended >= 4
                self.last_update_ts = time.time()
            else:
                self.hand_present = False
            time.sleep(0.002)

    def read_controls(self):
        return {
            "hand_present": self.hand_present,
            "x_norm": self.hand_x_norm,
            "pinch": self.pinch_active,
            "open_palm": self.open_palm,
            "stale": (time.time() - self.last_update_ts) > 0.5
        }

    def get_latest_frame(self):
        return self.last_frame.copy()

    def stop(self):
        self.running = False
        try: self.thread.join(timeout=1.0)
        except: pass
        self.hands.close()
        if self.cap: self.cap.release()

# ---------------------------
# Blink Detector (for replay)
# ---------------------------
class BlinkDetector:
    def __init__(self, cam_index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.running = True
        self.blink_timestamps = []
        self.double_blink = False
        threading.Thread(target=self._run, daemon=True).start()

    def _eye_aspect_ratio(self, lm, idx):
        p = np.array([(lm[i].x, lm[i].y) for i in idx])
        A = np.linalg.norm(p[1]-p[5])
        B = np.linalg.norm(p[2]-p[4])
        C = np.linalg.norm(p[0]-p[3])
        return (A + B) / (2.0 * C + 1e-6)

    def _run(self):
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        THRESH = 0.21
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            frame = cv2.flip(frame, 1)
            res = self.face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                left_EAR = self._eye_aspect_ratio(lm, LEFT_EYE)
                right_EAR = self._eye_aspect_ratio(lm, RIGHT_EYE)
                ear = (left_EAR + right_EAR) / 2.0
                if ear < THRESH:
                    now = time.time()
                    if (not self.blink_timestamps) or (now - self.blink_timestamps[-1] > 0.3):
                        self.blink_timestamps.append(now)
                        if len(self.blink_timestamps) >= 2 and (now - self.blink_timestamps[-2] < 1.0):
                            self.double_blink = True
            time.sleep(0.01)

    def check_double_blink(self):
        if self.double_blink:
            self.double_blink = False
            return True
        return False

    def stop(self):
        self.running = False
        try: self.cap.release()
        except: pass

# ---------------------------
# Neon Gesture Game
# ---------------------------
class NeonRunner:
    def __init__(self, width=960, height=540):
        pygame.init()
        pygame.display.set_caption("NEON RUNNER — Gesture + Blink Replay + Camera Feed")
        self.W, self.H = width, height
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 20)

        self.C_BG=(5,8,20); self.C_GRID=(0,180,255); self.C_PLAYER=(255,50,200)
        self.C_BULLET=(255,255,0); self.C_OBS=(0,255,180); self.C_TEXT=(180,220,255)

        self.player_x=self.W//2; self.player_y=self.H-80
        self.player_speed=10; self.boost_timer=0
        self.bullets=[]; self.obstacles=[]; self.spawn_timer=0
        self.score=0; self.game_over=False

        self.detector=GestureDetector(process_every_n_frames=2)
        self.blinker=BlinkDetector()
        self.grid_phase=0.0

    def spawn_obstacle(self):
        w=random.randint(30,80); h=random.randint(15,30)
        x=random.randint(0,self.W-w); y=-h
        speed=random.uniform(3.0,6.5)+(self.score/1000.0)
        self.obstacles.append({"x":x,"y":y,"w":w,"h":h,"speed":speed})

    def update(self, dt):
        if self.game_over: return
        c=self.detector.read_controls()
        if c["hand_present"] and not c["stale"]:
            target_x=int(c["x_norm"]*self.W)
            speed=self.player_speed*(1.8 if self.boost_timer>0 else 1.0)
            self.player_x+=(speed if target_x>self.player_x else -speed) if abs(target_x-self.player_x)>speed else (target_x-self.player_x)
            if c["pinch"] and len(self.bullets)<6:
                self.bullets.append([self.player_x,self.player_y-20])
            if c["open_palm"]: self.boost_timer=12
        if self.boost_timer>0: self.boost_timer-=1

        for b in self.bullets: b[1]-=12
        self.bullets=[b for b in self.bullets if b[1]>-20]

        self.spawn_timer+=dt
        if self.spawn_timer>max(0.35,1.0-(self.score/3000.0)):
            self.spawn_timer=0.0; self.spawn_obstacle()
        for o in self.obstacles: o["y"]+=o["speed"]
        self.obstacles=[o for o in self.obstacles if o["y"]<self.H+50]

        to_rm_o=set(); to_rm_b=set()
        for i,o in enumerate(self.obstacles):
            ox,oy,ow,oh=o["x"],o["y"],o["w"],o["h"]
            for j,b in enumerate(self.bullets):
                if ox<=b[0]<=ox+ow and oy<=b[1]<=oy+oh:
                    to_rm_o.add(i); to_rm_b.add(j); self.score+=25
        self.obstacles=[o for i,o in enumerate(self.obstacles) if i not in to_rm_o]
        self.bullets=[b for j,b in enumerate(self.bullets) if j not in to_rm_b]

        px,py=self.player_x,self.player_y
        for o in self.obstacles:
            if (o["x"]-20<=px<=o["x"]+o["w"]+20) and (o["y"]-10<=py<=o["y"]+o["h"]+10):
                self.game_over=True; break

        self.score+=int(dt*60)
        self.grid_phase+=dt*2.0

    def draw_grid(self):
        spacing=40
        for y in range(0,self.H,spacing):
            p=(y+int(self.grid_phase*80))%(self.H+spacing)
            pygame.draw.line(self.screen,self.C_GRID,(0,p),(self.W,p),1)
        for x in range(0,self.W,spacing):
            pygame.draw.line(self.screen,self.C_GRID,(x,0),(x,self.H),1)

    def draw_player(self):
        x,y=self.player_x,self.player_y
        pts=[(x,y-18),(x-14,y+14),(x+14,y+14)]
        pygame.draw.polygon(self.screen,self.C_PLAYER,pts,2)
        if self.boost_timer>0:
            glow=pygame.Surface((60,60),pygame.SRCALPHA)
            pygame.draw.circle(glow,(255,50,200,70),(30,30),28)
            self.screen.blit(glow,(x-30,y-30))

    def draw_ui(self):
        self.screen.blit(self.font.render(f"Score: {self.score}",True,self.C_TEXT),(12,10))

    def draw_debug(self,c):
        status=f"Hand:{'Y' if c['hand_present'] else 'N'}  Pinch:{'Y' if c['pinch'] else 'N'}  Palm:{'Y' if c['open_palm'] else 'N'}"
        self.screen.blit(self.font.render(status,True,(120,200,255)),(12,36))

    def draw_camera_feed(self):
        frame = self.detector.get_latest_frame()
        if frame is None or frame.size == 0: return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (200, 150))
        surf = pygame.surfarray.make_surface(np.rot90(frame))
        self.screen.blit(surf, (10, self.H - 160))

    def render(self):
        self.screen.fill(self.C_BG)
        self.draw_grid()
        for o in self.obstacles:
            pygame.draw.rect(self.screen,self.C_OBS,pygame.Rect(int(o["x"]),int(o["y"]),o["w"],o["h"]),2)
        for b in self.bullets:
            pygame.draw.circle(self.screen,self.C_BULLET,(int(b[0]),int(b[1])),4)
        self.draw_player()
        self.draw_ui()
        self.draw_camera_feed()

    def run(self):
        running=True
        while running:
            dt=self.clock.tick(60)/1000.0
            for e in pygame.event.get():
                if e.type==pygame.QUIT: running=False
                if e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE: running=False

            self.update(dt)
            self.render()
            c=self.detector.read_controls()
            self.draw_debug(c)

            if self.blinker.check_double_blink():
                if self.game_over:
                    self.bullets.clear()
                    self.obstacles.clear()
                    self.score=0
                    self.game_over=False
                    print("↻ Replay triggered by double blink")

            if self.game_over:
                over=self.font.render("GAME OVER — Blink twice to Replay",True,(255,120,120))
                self.screen.blit(over,(self.W//2-over.get_width()//2,self.H//2-12))

            pygame.display.flip()

        self.detector.stop()
        self.blinker.stop()
        pygame.quit()

if __name__=="__main__":
    NeonRunner().run()


