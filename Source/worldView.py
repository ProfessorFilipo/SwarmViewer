
##################################################################
####                 S W A R M    V I E W E R                 ####
##################################################################
#### part of the Beyond Visual Sight Drone Operation Project  ####
##################################################################
#### Prof. Filipo - github.com/ProfessorFilipo/SwarmViewer    ####
##################################################################

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import json
import math
import numpy as np
import os
import sys
import cv2


# ==========================================
# 1. Classe Drone (Atualizada para parâmetros orbitais)
# ==========================================

class Drone:
    def __init__(self, data, passo_global):
        self.id = data.get('id')
        self.pos = np.array(data.get('posicao_inicial'), dtype=float)
        self.pos_final = np.array(data.get('posicao_final'), dtype=float)
        self.safety_radius = data.get('area_seguranca', 1.0)
        self.trajectory_type = data.get('tipo_trajetoria', 'linear')
        self.dt = data.get('passo_tempo', passo_global)
        self.color = data.get('cor_rgb', [0, 1, 0])

        # --- NOVOS PARÂMETROS ORBITAIS ---
        # Se não houver no JSON, usa valores padrão (5.0 raio, 1.0 velocidade)
        self.orbit_radius = data.get('raio_orbita', 5.0)
        self.speed_factor = data.get('velocidade_orbital', 1.0)

        # Estado interno
        self.time_elapsed = 0
        self.start_pos = self.pos.copy()

    def update(self):
        self.time_elapsed += self.dt * 0.1

        if self.trajectory_type == 'linear':
            # Movimento Linear (A -> B)
            direction = self.pos_final - self.start_pos
            distance = np.linalg.norm(direction)
            if distance > 0:
                norm_dir = direction / distance
                move = norm_dir * self.time_elapsed
                if np.linalg.norm(move) < distance:
                    self.pos = self.start_pos + move
                else:
                    self.pos = self.pos_final

        elif self.trajectory_type == 'orbital':
            # Movimento Circular ao redor da Posição Inicial
            # O ângulo é afetado pela velocidade_orbital individual
            angle = self.time_elapsed * self.speed_factor

            self.pos[0] = self.start_pos[0] + math.cos(angle) * self.orbit_radius
            self.pos[2] = self.start_pos[2] + math.sin(angle) * self.orbit_radius

            # Movimento vertical suave (bobbing/flutuação) para dar "vida" ao drone
            self.pos[1] = self.start_pos[1] + math.sin(self.time_elapsed * 0.5) * 0.5

    def draw_path(self):
        glLineWidth(1)

        if self.trajectory_type == 'linear':
            # Linha Branca até o Destino
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_LINES)
            glVertex3f(self.pos[0], self.pos[1], self.pos[2])
            glVertex3f(self.pos_final[0], self.pos_final[1], self.pos_final[2])
            glEnd()

            # Marcador "X" no destino
            glPushMatrix()
            glTranslatef(self.pos_final[0], self.pos_final[1], self.pos_final[2])
            glColor3f(1.0, 0.0, 0.0)
            glBegin(GL_LINES)
            s = 0.5
            glVertex3f(-s, 0, -s);
            glVertex3f(s, 0, s)
            glVertex3f(s, 0, -s);
            glVertex3f(-s, 0, s)
            glEnd()
            glPopMatrix()

        elif self.trajectory_type == 'orbital':
            # Linha Amarela (Raio) ligando ao Centro da Órbita
            glColor3f(1.0, 1.0, 0.0)
            glBegin(GL_LINES)
            glVertex3f(self.pos[0], self.pos[1], self.pos[2])
            glVertex3f(self.start_pos[0], self.start_pos[1], self.start_pos[2])
            glEnd()

    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])

        # Corpo do Drone
        glColor3f(*self.color)
        quad = gluNewQuadric()
        gluSphere(quad, 0.5, 16, 16)

        # Área de Segurança
        glColor4f(1, 0, 0, 0.5)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        gluSphere(quad, self.safety_radius, 16, 16)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glPopMatrix()


# ==========================================
# 2. Classe Obstáculo
# ==========================================

class Obstacle:
    def __init__(self, data):
        self.id = data.get('id')
        self.pos = data.get('posicao')
        self.dims = data.get('dimensoes')
        self.transparency = data.get('transparencia', 0) / 100.0
        self.collisional = data.get('colisional', True)

    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glRotatef(-90, 1, 0, 0)

        alpha = 1.0 - self.transparency
        if self.collisional:
            glColor4f(0.6, 0.6, 0.6, alpha)  # Cinza metálico
        else:
            glColor4f(0.0, 0.5, 1.0, alpha)  # Azul holográfico

        quad = gluNewQuadric()
        gluCylinder(quad, self.dims['raio'], self.dims['raio'], self.dims['altura'], 24, 2)
        gluDisk(quad, 0, self.dims['raio'], 24, 1)
        glTranslatef(0, 0, self.dims['altura'])
        gluDisk(quad, 0, self.dims['raio'], 24, 1)
        glPopMatrix()


# ==========================================
# 3. Motor de Simulação (Dual Recorder)
# ==========================================

class SimulationEngine:
    def __init__(self, config_file):
        self.load_config(config_file)
        # Resolução HD (1280x720) - Bom equilíbrio
        self.display_size = (1280, 720)
        self.init_graphics()

        self.paused = True

        # Câmera Inicial
        self.cam_yaw = 45
        self.cam_pitch = 30
        self.cam_dist = 50
        self.cam_center = [0, 0, 0]
        self.mouse_drag = False

        # Sistema de Gravação
        self.recording = False
        self.recording_mode = None
        self.video_writer = None

    def load_config(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.global_dt = data['simulacao'].get('passo_tempo_global', 0.1)
        self.sky_limit = int(data['ambiente'].get('limite_ceu', 100))
        self.drones = [Drone(d, self.global_dt) for d in data.get('drones', [])]
        self.obstacles = [Obstacle(o) for o in data.get('obstaculos', [])]

    def init_graphics(self):
        pygame.init()

        # Configurações de Qualidade (Anti-Aliasing)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        self.screen = pygame.display.set_mode(self.display_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Plataforma Drone 3D - V3.0 Final")

        # Configuração da Lente
        glViewport(0, 0, self.display_size[0], self.display_size[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.display_size[0] / self.display_size[1]), 1.0, 1000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Suavização de Linhas
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glDisable(GL_LIGHTING)  # Estilo Flat/Neon

    def update_camera(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        rad_yaw = math.radians(self.cam_yaw)
        rad_pitch = math.radians(self.cam_pitch)

        cam_x = self.cam_center[0] + self.cam_dist * math.sin(rad_yaw) * math.cos(rad_pitch)
        cam_y = self.cam_center[1] + self.cam_dist * math.sin(rad_pitch)
        cam_z = self.cam_center[2] + self.cam_dist * math.cos(rad_yaw) * math.cos(rad_pitch)

        gluLookAt(cam_x, cam_y, cam_z,
                  self.cam_center[0], self.cam_center[1], self.cam_center[2],
                  0, 1, 0)

    def draw_environment(self):
        # Eixos
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(5, 0, 0)
        glColor3f(0, 1, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 5, 0)
        glColor3f(0, 0, 1);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 5)
        glEnd()

        # Solo (Roxo)
        glLineWidth(1)
        glColor3f(0.5, 0.0, 0.5)
        glBegin(GL_LINES)
        for i in range(-self.sky_limit, self.sky_limit + 1, 5):
            glVertex3f(-self.sky_limit, 0, i);
            glVertex3f(self.sky_limit, 0, i)
            glVertex3f(i, 0, -self.sky_limit);
            glVertex3f(i, 0, self.sky_limit)
        glEnd()

        # Céu (Azul Ciano)
        glColor3f(0.0, 0.8, 1.0)
        glBegin(GL_LINES)
        for i in range(-self.sky_limit, self.sky_limit + 1, 5):
            glVertex3f(-self.sky_limit, 20, i);
            glVertex3f(self.sky_limit, 20, i)
            glVertex3f(i, 20, -self.sky_limit);
            glVertex3f(i, 20, self.sky_limit)
        glEnd()

    # --- LÓGICA DE GRAVAÇÃO ---
    def start_recording(self, mode):
        if self.recording: self.stop_recording()

        self.recording = True
        self.recording_mode = mode

        if mode == 'LOW':
            filename = "video_low_quality.avi"
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        elif mode == 'HIGH':
            filename = "video_high_quality.mp4"
            # Tenta codec MP4V, se falhar usa MJPG
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            except:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        print(f"REC [{mode}]: Iniciando gravação em {filename}...")
        self.video_writer = cv2.VideoWriter(filename, fourcc, 60.0, self.display_size)

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        print("REC: Gravação finalizada.")
        self.recording_mode = None

    def capture_frame_if_active(self):
        # PERFORMANCE: Retorna imediatamente se não estiver gravando
        if not self.recording or self.video_writer is None:
            return

        # Captura pixels
        buffer = glReadPixels(0, 0, self.display_size[0], self.display_size[1], GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(buffer, dtype=np.uint8)
        image = image.reshape((self.display_size[1], self.display_size[0], 3))
        image = np.flipud(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.video_writer.write(image)

    def run(self):
        clock = pygame.time.Clock()
        running = True
        print("--- Controles ---")
        print("ESPAÇO: Play/Pause")
        print("MOUSE:  Câmera")
        print("V:      Gravar Low Quality (.avi)")
        print("H:      Gravar High Quality (.mp4)")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: self.paused = not self.paused
                    if event.key == pygame.K_r:
                        self.cam_dist = 50;
                        self.cam_pitch = 30;
                        self.cam_center = [0, 0, 0]
                    if event.key == pygame.K_t:
                        self.cam_pitch = 89;
                        self.cam_yaw = 0

                    # Controles de Gravação
                    if event.key == pygame.K_v:
                        if self.recording and self.recording_mode == 'LOW':
                            self.stop_recording()
                        else:
                            self.start_recording('LOW')

                    if event.key == pygame.K_h:
                        if self.recording and self.recording_mode == 'HIGH':
                            self.stop_recording()
                        else:
                            self.start_recording('HIGH')

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: self.mouse_drag = True
                    if event.button == 4: self.cam_dist -= 2
                    if event.button == 5: self.cam_dist += 2
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1: self.mouse_drag = False
                if event.type == pygame.MOUSEMOTION:
                    if self.mouse_drag:
                        self.cam_yaw -= event.rel[0] * 0.5
                        self.cam_pitch += event.rel[1] * 0.5
                        self.cam_pitch = max(1.0, min(89.0, self.cam_pitch))

            if not self.paused:
                for d in self.drones:
                    d.update()

            # Renderização
            glClearColor(0.05, 0.05, 0.2, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.update_camera()
            self.draw_environment()

            for o in self.obstacles: o.draw()
            for d in self.drones:
                d.draw_path()
                d.draw()

            # Captura (sem impacto se desligado)
            self.capture_frame_if_active()

            # Interface Visual (Título)
            rec_text = f"[● REC {self.recording_mode}]" if self.recording else ""
            status = "PAUSADO" if self.paused else "PLAY"
            pygame.display.set_caption(f"Drone Sim 3D | {status} | FPS: {int(clock.get_fps())} {rec_text}")

            pygame.display.flip()
            clock.tick(60)

        if self.video_writer:
            self.video_writer.release()

        pygame.quit()
        sys.exit()


# ==========================================
# 4. Geração Automática de Config (5 Drones)
# ==========================================
if __name__ == "__main__":
    if not os.path.exists("config.json"):
        print("Gerando nova configuração orbital...")
        config_data = {
            "simulacao": {
                "passo_tempo_global": 0.2,
                "duracao_maxima": 2000
            },
            "ambiente": {
                "limite_ceu": 60,
                "limite_solo": 60
            },
            "obstaculos": [
                {
                    "id": "torre_central",
                    "posicao": [0, 0, 0],
                    "dimensoes": {"altura": 15, "raio": 2},
                    "colisional": True,
                    "transparencia": 40
                }
            ],
            # 5 Drones Orbitais, raios crescentes, velocidades variadas
            "drones": [
                {
                    "id": "drone_1", "tipo_trajetoria": "orbital", "cor_rgb": [1.0, 0.0, 0.0],
                    "posicao_inicial": [0, 5, 0], "posicao_final": [0, 0, 0], "area_seguranca": 1.0,
                    "raio_orbita": 4.0, "velocidade_orbital": 1.5
                },
                {
                    "id": "drone_2", "tipo_trajetoria": "orbital", "cor_rgb": [0.0, 1.0, 0.0],
                    "posicao_inicial": [0, 6, 0], "posicao_final": [0, 0, 0], "area_seguranca": 1.0,
                    "raio_orbita": 7.0, "velocidade_orbital": 1.0
                },
                {
                    "id": "drone_3", "tipo_trajetoria": "orbital", "cor_rgb": [0.0, 0.0, 1.0],
                    "posicao_inicial": [0, 7, 0], "posicao_final": [0, 0, 0], "area_seguranca": 1.0,
                    "raio_orbita": 10.0, "velocidade_orbital": 0.8
                },
                {
                    "id": "drone_4", "tipo_trajetoria": "orbital", "cor_rgb": [1.0, 1.0, 0.0],
                    "posicao_inicial": [0, 8, 0], "posicao_final": [0, 0, 0], "area_seguranca": 1.0,
                    "raio_orbita": 13.0, "velocidade_orbital": -0.5
                },
                {
                    "id": "drone_5", "tipo_trajetoria": "orbital", "cor_rgb": [1.0, 0.0, 1.0],
                    "posicao_inicial": [0, 9, 0], "posicao_final": [0, 0, 0], "area_seguranca": 1.2,
                    "raio_orbita": 16.0, "velocidade_orbital": -1.2
                }
            ]
        }
        with open("config.json", 'w') as f:
            json.dump(config_data, f, indent=4)

    app = SimulationEngine("config.json")
    app.run()
