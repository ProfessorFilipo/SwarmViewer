import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import json
import math
import numpy as np
import os
import sys


# ==========================================
# 1. Classes de Lógica
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
        elif self.trajectory_type == 'orbital':
            radius = 5.0
            self.pos[0] = self.start_pos[0] + math.cos(self.time_elapsed) * radius
            self.pos[2] = self.start_pos[2] + math.sin(self.time_elapsed) * radius
            self.pos[1] = self.start_pos[1] + (self.time_elapsed * 0.2)

    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])

        # Drone (Cubo pequeno para garantir visibilidade se esfera falhar)
        glColor3f(*self.color)
        quad = gluNewQuadric()
        gluSphere(quad, 0.5, 16, 16)

        # Área de segurança
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
            glColor4f(0.7, 0.7, 0.7, alpha)
        else:
            glColor4f(0.0, 0.5, 1.0, alpha)

        quad = gluNewQuadric()
        gluCylinder(quad, self.dims['raio'], self.dims['raio'], self.dims['altura'], 20, 2)
        glPopMatrix()


# ==========================================
# 2. Motor Gráfico (CORRIGIDO)
# ==========================================

class SimulationEngine:
    def __init__(self, config_file):
        self.load_config(config_file)
        self.display_size = (1024, 768)
        self.init_graphics()

        self.paused = True
        self.cam_yaw = 45  # Ângulo inicial para ver algo de esquina
        self.cam_pitch = 30
        self.cam_dist = 40
        self.cam_center = [0, 0, 0]
        self.mouse_drag = False

    def load_config(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.global_dt = data['simulacao'].get('passo_tempo_global', 0.1)
        self.sky_limit = int(data['ambiente'].get('limite_ceu', 100))
        self.drones = [Drone(d, self.global_dt) for d in data.get('drones', [])]
        self.obstacles = [Obstacle(o) for o in data.get('obstaculos', [])]

    def init_graphics(self):
        pygame.init()
        # Configurações de Buffer
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        self.screen = pygame.display.set_mode(self.display_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Drone Sim 3D - FIXED MATRIX")

        # --- CORREÇÃO DE MATRIZ (O Segredo) ---
        glViewport(0, 0, self.display_size[0], self.display_size[1])

        # 1. Configura a Lente (PROJECTION)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.display_size[0] / self.display_size[1]), 1.0, 1000.0)

        # 2. Volta para o modo de Objetos (MODELVIEW)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)  # Cores puras

    def update_camera(self):
        # Garante que estamos mexendo na visualização, não na lente
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

    def draw_debug_elements(self):
        """ Desenha eixos gigantes e chão para garantir que algo apareça """
        glLineWidth(2)
        glBegin(GL_LINES)

        # Eixo X (Vermelho)
        glColor3f(1, 0, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(20, 0, 0)
        # Eixo Y (Verde)
        glColor3f(0, 1, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 20, 0)
        # Eixo Z (Azul)
        glColor3f(0, 0, 1);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 20)

        # Grid simples (Branco)
        glColor3f(0.3, 0.3, 0.3)
        for i in range(-20, 21, 5):
            glVertex3f(i, 0, -20);
            glVertex3f(i, 0, 20)
            glVertex3f(-20, 0, i);
            glVertex3f(20, 0, i)
        glEnd()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        print("Iniciando Renderização...")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: self.paused = not self.paused
                    if event.key == pygame.K_r: self.cam_dist = 40

                # Input Mouse
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

            if not self.paused:
                for d in self.drones: d.update()

            # Render
            glClearColor(0.1, 0.1, 0.2, 1.0)  # Fundo Azul Escuro
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # 1. Posiciona Câmera
            self.update_camera()

            # 2. Desenha Mundo
            self.draw_debug_elements()

            # 3. Desenha Objetos
            for o in self.obstacles: o.draw()
            for d in self.drones: d.draw()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    # Garante config
    if not os.path.exists("config.json"):
        dummy_config = {
            "simulacao": {"passo_tempo_global": 0.2},
            "ambiente": {"limite_ceu": 50},
            "drones": [
                {"id": "d1", "posicao_inicial": [0, 5, 0], "posicao_final": [10, 5, 10], "area_seguranca": 1.5,
                 "cor_rgb": [1, 1, 0], "tipo_trajetoria": "orbital"}
            ],
            "obstaculos": []
        }
        with open("config.json", 'w') as f:
            json.dump(dummy_config, f, indent=4)

    app = SimulationEngine("config.json")
    app.run()
