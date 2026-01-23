
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


# ==========================================
# 1. Classes de Entidades
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

        # Estado interno
        self.time_elapsed = 0
        self.start_pos = self.pos.copy()

    def update(self):
        self.time_elapsed += self.dt * 0.1  # Fator de velocidade

        if self.trajectory_type == 'linear':
            direction = self.pos_final - self.start_pos
            distance = np.linalg.norm(direction)
            if distance > 0:
                norm_dir = direction / distance
                move = norm_dir * self.time_elapsed
                # Verifica se chegou (não ultrapassar)
                if np.linalg.norm(move) < distance:
                    self.pos = self.start_pos + move
                else:
                    self.pos = self.pos_final

        elif self.trajectory_type == 'orbital':
            # Exemplo de lógica orbital (Raio 5 em volta do eixo Y relativo ao inicio)
            radius = 5.0
            self.pos[0] = self.start_pos[0] + math.cos(self.time_elapsed) * radius
            self.pos[2] = self.start_pos[2] + math.sin(self.time_elapsed) * radius
            # Sobe levemente em espiral
            self.pos[1] = self.start_pos[1] + (self.time_elapsed * 0.2)

    def draw_path(self):
        """ Desenha a linha de trajetória prevista """
        glLineWidth(1)
        glColor3f(1.0, 1.0, 1.0)  # Linha Branca

        glBegin(GL_LINES)
        glVertex3f(self.pos[0], self.pos[1], self.pos[2])  # De onde está
        glVertex3f(self.pos_final[0], self.pos_final[1], self.pos_final[2])  # Para onde vai
        glEnd()

        # Desenha um pequeno marcador no destino
        glPushMatrix()
        glTranslatef(self.pos_final[0], self.pos_final[1], self.pos_final[2])
        glColor3f(1.0, 0.0, 0.0)  # Vermelho
        glBegin(GL_LINES)
        s = 0.5  # Tamanho do X
        glVertex3f(-s, 0, -s);
        glVertex3f(s, 0, s)
        glVertex3f(s, 0, -s);
        glVertex3f(-s, 0, s)
        glEnd()
        glPopMatrix()

    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])

        # Corpo do Drone (Esfera sólida)
        glColor3f(*self.color)
        quad = gluNewQuadric()
        gluSphere(quad, 0.5, 16, 16)

        # Área de Segurança (Wireframe)
        glColor4f(1, 0, 0, 0.5)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        gluSphere(quad, self.safety_radius, 16, 16)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glPopMatrix()


class Obstacle:
    def __init__(self, data):
        self.id = data.get('id')
        self.pos = data.get('posicao')
        self.dims = data.get('dimensoes')  # altura, raio
        self.transparency = data.get('transparencia', 0) / 100.0
        self.collisional = data.get('colisional', True)

    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glRotatef(-90, 1, 0, 0)  # Coloca o cilindro em pé

        alpha = 1.0 - self.transparency
        if self.collisional:
            glColor4f(0.6, 0.6, 0.6, alpha)  # Cinza
        else:
            glColor4f(0.0, 0.5, 1.0, alpha)  # Azul

        quad = gluNewQuadric()
        # Base, Topo, Altura, Fatias, Pilhas
        gluCylinder(quad, self.dims['raio'], self.dims['raio'], self.dims['altura'], 20, 2)
        # Tampas
        gluDisk(quad, 0, self.dims['raio'], 20, 1)
        glTranslatef(0, 0, self.dims['altura'])
        gluDisk(quad, 0, self.dims['raio'], 20, 1)

        glPopMatrix()


# ==========================================
# 2. Motor de Simulação
# ==========================================

class SimulationEngine:
    def __init__(self, config_file):
        self.load_config(config_file)
        self.display_size = (1024, 768)
        self.init_graphics()

        # Estado
        self.paused = True

        # Câmera
        self.cam_yaw = 45
        self.cam_pitch = 30
        self.cam_dist = 50
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
        # Configurações para robustez visual
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        self.screen = pygame.display.set_mode(self.display_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Plataforma Drone 3D - V1.1")

        # Configuração de Matriz de Projeção (A "Lente")
        glViewport(0, 0, self.display_size[0], self.display_size[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.display_size[0] / self.display_size[1]), 1.0, 1000.0)

        # Volta para ModelView para desenhar objetos
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_LIGHTING)  # Estilo Neon/Flat

    def update_camera(self):
        # Reinicia a matriz de visualização a cada frame
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

    def draw_grid(self, y_level, color, size=50, step=5):
        """ Desenha grids (chão ou teto) """
        glLineWidth(1)
        glColor3f(*color)
        glBegin(GL_LINES)

        for i in range(-size, size + 1, step):
            # Linhas eixo X
            glVertex3f(-size, y_level, i)
            glVertex3f(size, y_level, i)
            # Linhas eixo Z
            glVertex3f(i, y_level, -size)
            glVertex3f(i, y_level, size)
        glEnd()

    def draw_environment(self):
        # Eixos Centrais (Referência)
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(5, 0, 0)  # X
        glColor3f(0, 1, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 5, 0)  # Y
        glColor3f(0, 0, 1);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 5)  # Z
        glEnd()

        # Chão (Roxo estilo anos 80)
        self.draw_grid(y_level=0, color=(0.5, 0.0, 0.5), size=self.sky_limit)

        # Teto/Céu (Ciano)
        self.draw_grid(y_level=20, color=(0.0, 0.8, 1.0), size=self.sky_limit)

    def run(self):
        clock = pygame.time.Clock()
        running = True
        print("Simulação Iniciada. Use MOUSE para Câmera, ESPAÇO para Pause/Play.")

        while running:
            # 1. Inputs
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Teclado
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: self.paused = not self.paused
                    if event.key == pygame.K_r:  # Reset Camera
                        self.cam_dist = 50;
                        self.cam_pitch = 30;
                        self.cam_center = [0, 0, 0]
                    if event.key == pygame.K_t:  # Top View
                        self.cam_pitch = 89;
                        self.cam_yaw = 0

                # Mouse
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
                        # Trava de segurança para câmera não inverter
                        self.cam_pitch = max(1.0, min(89.0, self.cam_pitch))

            # 2. Física / Lógica
            if not self.paused:
                for d in self.drones:
                    d.update()

            # 3. Renderização
            # Limpa tela com azul escuro
            glClearColor(0.05, 0.05, 0.2, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.update_camera()
            self.draw_environment()

            # Desenha Obstáculos
            for o in self.obstacles:
                o.draw()

            # Desenha Drones e suas trajetórias
            for d in self.drones:
                d.draw_path()  # Nova linha de trajetória
                d.draw()

            # Update Janela
            status = "PAUSADO" if self.paused else "PLAY"
            pygame.display.set_caption(f"Simulador Drone 3D | {status} | FPS: {int(clock.get_fps())}")

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    # Garante arquivo de config
    if not os.path.exists("config.json"):
        dummy_config = {
            "simulacao": {"passo_tempo_global": 0.2},
            "ambiente": {"limite_ceu": 40},
            "drones": [
                {
                    "id": "drone_linear",
                    "posicao_inicial": [-10, 5, -10],
                    "posicao_final": [10, 10, 10],
                    "area_seguranca": 1.5,
                    "cor_rgb": [0, 1, 0],
                    "tipo_trajetoria": "linear"
                },
                {
                    "id": "drone_orbital",
                    "posicao_inicial": [5, 5, 5],
                    "posicao_final": [0, 0, 0],
                    "area_seguranca": 1.2,
                    "cor_rgb": [1, 0.5, 0],
                    "tipo_trajetoria": "orbital"
                }
            ],
            "obstaculos": [
                {"id": "predio", "posicao": [0, 0, 0], "dimensoes": {"altura": 8, "raio": 2}, "transparencia": 30}
            ]
        }
        with open("config.json", 'w') as f:
            json.dump(dummy_config, f, indent=4)

    app = SimulationEngine("config.json")
    app.run()

