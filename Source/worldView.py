
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
# 1. Classes de Entidades (Inalteradas)
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
        self.time_elapsed = 0
        self.start_pos = self.pos.copy()

    def update(self):
        self.time_elapsed += self.dt * 0.1
        if self.trajectory_type == 'linear':
            direction = self.pos_final - self.start_pos
            distance = np.linalg.norm(direction)
            if distance > 0:
                norm_dir = direction / distance
                move = norm_dir * self.time_elapsed
                if np.linalg.norm(move) < distance:
                    self.pos = self.start_pos + move
                else:
                    self.pos = self.pos_final
        #elif self.trajectory_type == 'orbital':
        #    radius = 5.0
        #    self.pos[0] = self.start_pos[0] + math.cos(self.time_elapsed) * radius
        #    self.pos[2] = self.start_pos[2] + math.sin(self.time_elapsed) * radius
        #    self.pos[1] = self.start_pos[1] + (self.time_elapsed * 0.2)

        elif self.trajectory_type == 'orbital':
            radius = 5.0

            # Lógica Horizontal (X e Z) - Continua igual
            self.pos[0] = self.start_pos[0] + math.cos(self.time_elapsed) * radius
            self.pos[2] = self.start_pos[2] + math.sin(self.time_elapsed) * radius

            # Lógica Vertical (Y) - AGORA USA A POSIÇÃO FINAL
            # Se a altura atual for menor que a altura final, continua subindo
            if self.pos[1] < self.pos_final[1]:
                self.pos[1] = self.start_pos[1] + (self.time_elapsed * 0.5)

                # Garante que não ultrapasse a altura final
                if self.pos[1] > self.pos_final[1]:
                    self.pos[1] = self.pos_final[1]
            else:
                # Se já chegou na altura, mantém fixo na altura final
                self.pos[1] = self.pos_final[1]

    def draw_path(self):
        glLineWidth(1)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(self.pos[0], self.pos[1], self.pos[2])
        glVertex3f(self.pos_final[0], self.pos_final[1], self.pos_final[2])
        glEnd()

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

    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glColor3f(*self.color)
        quad = gluNewQuadric()
        gluSphere(quad, 0.5, 16, 16)

        glColor4f(1, 0, 0, 0.5)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        gluSphere(quad, self.safety_radius, 16, 16)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPopMatrix()


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
            glColor4f(0.6, 0.6, 0.6, alpha)
        else:
            glColor4f(0.0, 0.5, 1.0, alpha)
        quad = gluNewQuadric()
        gluCylinder(quad, self.dims['raio'], self.dims['raio'], self.dims['altura'], 20, 2)
        gluDisk(quad, 0, self.dims['raio'], 20, 1)
        glTranslatef(0, 0, self.dims['altura'])
        gluDisk(quad, 0, self.dims['raio'], 20, 1)
        glPopMatrix()


# ==========================================
# 2. Motor de Simulação (Dual Recorder)
# ==========================================

class SimulationEngine:
    def __init__(self, config_file):
        self.load_config(config_file)
        # Resolução Padrão (pode aumentar se quiser HD nativo na tela)
        self.display_size = (1280, 720)
        self.init_graphics()

        self.paused = True

        # Câmera
        self.cam_yaw = 45
        self.cam_pitch = 30
        self.cam_dist = 50
        self.cam_center = [0, 0, 0]
        self.mouse_drag = False

        # --- SISTEMA DE GRAVAÇÃO HÍBRIDO ---
        self.recording = False
        self.recording_mode = None  # 'LOW' ou 'HIGH'
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

        # Ativa Suavização (Anti-Aliasing) globalmente para ficar bonito sempre
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)

        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        self.screen = pygame.display.set_mode(self.display_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Plataforma Drone 3D")

        glViewport(0, 0, self.display_size[0], self.display_size[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.display_size[0] / self.display_size[1]), 1.0, 1000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Configurações de linha suave
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glDisable(GL_LIGHTING)

    def update_camera(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        rad_yaw = math.radians(self.cam_yaw)
        rad_pitch = math.radians(self.cam_pitch)
        cam_x = self.cam_center[0] + self.cam_dist * math.sin(rad_yaw) * math.cos(rad_pitch)
        cam_y = self.cam_center[1] + self.cam_dist * math.sin(rad_pitch)
        cam_z = self.cam_center[2] + self.cam_dist * math.cos(rad_yaw) * math.cos(rad_pitch)
        gluLookAt(cam_x, cam_y, cam_z, self.cam_center[0], self.cam_center[1], self.cam_center[2], 0, 1, 0)

    def draw_environment(self):
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

        # Grid Solo
        glLineWidth(1)
        glColor3f(0.5, 0.0, 0.5)
        glBegin(GL_LINES)
        for i in range(-self.sky_limit, self.sky_limit + 1, 5):
            glVertex3f(-self.sky_limit, 0, i);
            glVertex3f(self.sky_limit, 0, i)
            glVertex3f(i, 0, -self.sky_limit);
            glVertex3f(i, 0, self.sky_limit)
        glEnd()

        # Grid Céu
        glColor3f(0.0, 0.8, 1.0)
        glBegin(GL_LINES)
        for i in range(-self.sky_limit, self.sky_limit + 1, 5):
            glVertex3f(-self.sky_limit, 20, i);
            glVertex3f(self.sky_limit, 20, i)
            glVertex3f(i, 20, -self.sky_limit);
            glVertex3f(i, 20, self.sky_limit)
        glEnd()

    # --- GERENCIADOR DE GRAVAÇÃO ---
    def start_recording(self, mode):
        # Se já estiver gravando, para a anterior primeiro
        if self.recording:
            self.stop_recording()

        self.recording = True
        self.recording_mode = mode

        if mode == 'LOW':
            # Modo V: Codec MJPG (Rápido, Baixa Qualidade, Arquivo AVI)
            filename = "video_low_quality.avi"
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            print(f"REC [LOW]: Iniciando gravação rápida em {filename}")

        elif mode == 'HIGH':
            # Modo H: Codec MP4V (Lento, Alta Qualidade, Arquivo MP4)
            filename = "video_high_quality.mp4"
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            except:
                print("Codec mp4v não encontrado, usando fallback...")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            print(f"REC [HIGH]: Iniciando gravação HD em {filename}")

        self.video_writer = cv2.VideoWriter(filename, fourcc, 60.0, self.display_size)

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        print(f"REC [{self.recording_mode}]: Gravação salva e finalizada.")
        self.recording_mode = None

    def capture_frame_if_active(self):
        # Check de performance: Se não estiver gravando, sai imediatamente da função
        if not self.recording or self.video_writer is None:
            return

        # Captura pixels
        buffer = glReadPixels(0, 0, self.display_size[0], self.display_size[1], GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(buffer, dtype=np.uint8)
        image = image.reshape((self.display_size[1], self.display_size[0], 3))
        image = np.flipud(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Escreve no disco
        self.video_writer.write(image)

    def run(self):
        clock = pygame.time.Clock()
        running = True
        print("--- Controles ---")
        print("ESPAÇO: Play/Pause")
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

                    # --- CONTROLE DUPLO DE GRAVAÇÃO ---
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

            glClearColor(0.05, 0.05, 0.2, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.update_camera()
            self.draw_environment()

            for o in self.obstacles: o.draw()
            for d in self.drones:
                d.draw_path()
                d.draw()

            # Captura sem impacto se estiver desligado
            self.capture_frame_if_active()

            # Feedback Visual na Janela
            rec_status = ""
            if self.recording:
                rec_status = f"[● REC {self.recording_mode}]"

            status = "PAUSADO" if self.paused else "PLAY"
            pygame.display.set_caption(f"Drone Sim | {status} | FPS: {int(clock.get_fps())} {rec_status}")

            pygame.display.flip()
            clock.tick(60)

        if self.video_writer:
            self.video_writer.release()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    if not os.path.exists("config.json"):
        dummy_config = {
            "simulacao": {"passo_tempo_global": 0.2},
            "ambiente": {"limite_ceu": 40},
            "drones": [
                {"id": "d1", "posicao_inicial": [-10, 5, -10], "posicao_final": [10, 10, 10], "area_seguranca": 2.0,
                 "cor_rgb": [0, 1, 0], "tipo_trajetoria": "linear"},
                {"id": "d2", "posicao_inicial": [5, 8, 5], "posicao_final": [0, 0, 0], "area_seguranca": 1.5,
                 "cor_rgb": [1, 0.5, 0], "tipo_trajetoria": "orbital"}
            ],
            "obstaculos": [
                {"id": "obs1", "posicao": [0, 0, 0], "dimensoes": {"altura": 10, "raio": 2}, "transparencia": 20,
                 "colisional": True}
            ]
        }
        with open("config.json", 'w') as f:
            json.dump(dummy_config, f, indent=4)

    app = SimulationEngine("config.json")
    app.run()

