
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
# 1. Classe Drone (Física e Colisão)
# ==========================================

class Drone:
    def __init__(self, data, passo_global):
        self.id = data.get('id')
        self.pos = np.array(data.get('posicao_inicial'), dtype=float)
        self.pos_final = np.array(data.get('posicao_final'), dtype=float)
        self.safety_radius = data.get('area_seguranca', 1.0)
        self.trajectory_type = data.get('tipo_trajetoria', 'linear')
        self.dt = data.get('passo_tempo', passo_global)
        self.original_color = data.get('cor_rgb', [0, 1, 0])

        # Parâmetros Orbitais
        self.orbit_radius = data.get('raio_orbita', 5.0)
        self.speed_factor = data.get('velocidade_orbital', 1.0)

        # --- HIT BOX & ESTADOS ---
        self.physical_radius = 0.5  # Hitbox física fixa conforme desenho (gluSphere 0.5)

        self.crashed = False  # Se True, congela
        self.warning_timer = 0  # Contador de frames para piscar (60 = 1 seg)

        self.time_elapsed = 0
        self.start_pos = self.pos.copy()

    def update(self):
        # Se bateu, congela a física
        if self.crashed:
            return

        self.time_elapsed += self.dt * 0.1

        # Decrementa timer de alerta se houver
        if self.warning_timer > 0:
            self.warning_timer -= 1

        # Lógica de Movimento
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
            angle = self.time_elapsed * self.speed_factor
            self.pos[0] = self.start_pos[0] + math.cos(angle) * self.orbit_radius
            self.pos[2] = self.start_pos[2] + math.sin(angle) * self.orbit_radius
            self.pos[1] = self.start_pos[1] + math.sin(self.time_elapsed * 0.5) * 0.5

    def draw_path(self):
        if self.crashed: return  # Não desenha linha se bateu

        glLineWidth(1)
        if self.trajectory_type == 'linear':
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_LINES)
            glVertex3f(self.pos[0], self.pos[1], self.pos[2])
            glVertex3f(self.pos_final[0], self.pos_final[1], self.pos_final[2])
            glEnd()
            # X no destino
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
            glColor3f(1.0, 1.0, 0.0)
            glBegin(GL_LINES)
            glVertex3f(self.pos[0], self.pos[1], self.pos[2])
            glVertex3f(self.start_pos[0], self.start_pos[1], self.start_pos[2])
            glEnd()

    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])

        # 1. Desenho do Corpo do Drone
        quad = gluNewQuadric()

        if self.crashed:
            # Modo Fantasma: Cinza, Transparente
            glColor4f(0.8, 0.8, 0.8, 0.3)
        else:
            # Cor Normal
            glColor3f(*self.original_color)

        gluSphere(quad, self.physical_radius, 16, 16)

        # 2. Desenho da Área de Segurança (Se não crashou)
        if not self.crashed:
            # Verifica se está em alerta (Piscando)
            if self.warning_timer > 0:
                # Pisca a cada 10 frames (Vermelho / Verde Brilhante)
                if (self.warning_timer // 10) % 2 == 0:
                    glColor4f(1.0, 0.0, 0.0, 0.6)  # Vermelho
                else:
                    glColor4f(0.0, 1.0, 0.0, 0.6)  # Verde

                # Linha mais grossa no alerta
                glLineWidth(2)
            else:
                # Normal (Vermelho transparente leve)
                glColor4f(1, 0, 0, 0.2)
                glLineWidth(1)

            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            gluSphere(quad, self.safety_radius, 16, 16)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glPopMatrix()


# ==========================================
# 2. Classe Obstáculo (Mantida)
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
            glColor4f(0.6, 0.6, 0.6, alpha)
        else:
            glColor4f(0.0, 0.5, 1.0, alpha)
        quad = gluNewQuadric()
        gluCylinder(quad, self.dims['raio'], self.dims['raio'], self.dims['altura'], 24, 2)
        gluDisk(quad, 0, self.dims['raio'], 24, 1)
        glTranslatef(0, 0, self.dims['altura'])
        gluDisk(quad, 0, self.dims['raio'], 24, 1)
        glPopMatrix()


# ==========================================
# 3. Motor de Simulação
# ==========================================

class SimulationEngine:
    def __init__(self, config_file):
        self.load_config(config_file)
        self.display_size = (1280, 720)
        self.init_graphics()

        self.paused = True
        self.cam_yaw = 45
        self.cam_pitch = 30
        self.cam_dist = 50
        self.cam_center = [0, 0, 0]
        self.mouse_drag = False

        self.recording = False
        self.recording_mode = None
        self.video_writer = None

        # Log de Colisões
        self.collision_log = []  # Lista de strings
        self.sim_time = 0.0  # Tempo total simulado

    def load_config(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.global_dt = data['simulacao'].get('passo_tempo_global', 0.1)
        self.sky_limit = int(data['ambiente'].get('limite_ceu', 100))
        self.drones = [Drone(d, self.global_dt) for d in data.get('drones', [])]
        self.obstacles = [Obstacle(o) for o in data.get('obstaculos', [])]

    def init_graphics(self):
        pygame.init()
        pygame.font.init()  # Inicializa fontes
        self.font = pygame.font.SysFont('Arial', 18)

        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        self.screen = pygame.display.set_mode(self.display_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Plataforma Drone 3D - V4.0 Collisions")

        glViewport(0, 0, self.display_size[0], self.display_size[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.display_size[0] / self.display_size[1]), 1.0, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
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

        glLineWidth(1)
        glColor3f(0.5, 0.0, 0.5)
        glBegin(GL_LINES)
        for i in range(-self.sky_limit, self.sky_limit + 1, 5):
            glVertex3f(-self.sky_limit, 0, i);
            glVertex3f(self.sky_limit, 0, i)
            glVertex3f(i, 0, -self.sky_limit);
            glVertex3f(i, 0, self.sky_limit)
        glEnd()

        glColor3f(0.0, 0.8, 1.0)
        glBegin(GL_LINES)
        for i in range(-self.sky_limit, self.sky_limit + 1, 5):
            glVertex3f(-self.sky_limit, 20, i);
            glVertex3f(self.sky_limit, 20, i)
            glVertex3f(i, 20, -self.sky_limit);
            glVertex3f(i, 20, self.sky_limit)
        glEnd()

    # --- LÓGICA DE COLISÃO ---
    def check_collisions(self):
        """ Verifica colisões N x N entre drones """
        num_drones = len(self.drones)
        for i in range(num_drones):
            for j in range(i + 1, num_drones):
                d1 = self.drones[i]
                d2 = self.drones[j]

                # Se algum já estiver crashado, ignora interações físicas
                if d1.crashed or d2.crashed:
                    continue

                # Calcula Distância Euclidiana
                dist = np.linalg.norm(d1.pos - d2.pos)

                # 1. Checa Colisão Física (CRASH)
                # Raio físico A (0.5) + Raio físico B (0.5) = 1.0
                limit_crash = d1.physical_radius + d2.physical_radius

                if dist < limit_crash:
                    d1.crashed = True
                    d2.crashed = True
                    msg = f"[{self.sim_time:.1f}s] CRASHED: {d1.id} <-> {d2.id}"
                    print(msg)
                    self.collision_log.append(msg)
                    continue  # Se crashou, não precisa checar warning

                # 2. Checa Colisão de Área (WARNING)
                limit_warning = d1.safety_radius + d2.safety_radius

                if dist < limit_warning:
                    # Ativa modo de alerta por 1 segundo (60 frames)
                    d1.warning_timer = 60
                    d2.warning_timer = 60

                    # Adiciona ao log apenas se não for spam (opcional, aqui logamos tudo)
                    # Para evitar spam no log, poderiamos checar se timer estava zerado
                    # msg = f"[{self.sim_time:.1f}s] ALERT: {d1.id} near {d2.id}"
                    # if msg not in self.collision_log[-1:]: # Evita duplicata imediata
                    #    self.collision_log.append(msg)

    # --- RENDERIZAÇÃO DE TEXTO NA TELA (UI) ---
    def draw_text_overlay(self):
        """ Desenha o log de colisões como uma sobreposição 2D """
        # Muda para projeção ortogonal (2D)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.display_size[0], 0, self.display_size[1])

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)  # Texto deve ficar por cima de tudo

        # Desenha o Log (Últimas 10 mensagens)
        y_offset = self.display_size[1] - 20
        glColor3f(1, 1, 1)  # Texto branco

        # Título
        self.render_text_gl("LOG DE EVENTOS:", 10, y_offset)
        y_offset -= 20

        # Itens (mostra os últimos 10)
        for msg in self.collision_log[-10:]:
            color = (1, 0, 0) if "CRASHED" in msg else (1, 1, 0)
            glColor3f(*color)
            self.render_text_gl(msg, 10, y_offset)
            y_offset -= 20

        # Restaura estado 3D
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def render_text_gl(self, text, x, y):
        """ Converte texto Pygame para Textura OpenGL (Simples) """
        text_surface = self.font.render(text, True, (255, 255, 255), (0, 0, 0))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        w, h = text_surface.get_size()

        glRasterPos2i(x, y - h)  # Posiciona bitmap
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    # --- LÓGICA DE GRAVAÇÃO (Mantida) ---
    def start_recording(self, mode):
        if self.recording: self.stop_recording()
        self.recording = True
        self.recording_mode = mode
        if mode == 'LOW':
            filename = "video_low_quality.avi"
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        elif mode == 'HIGH':
            filename = "video_high_quality.mp4"
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            except:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(filename, fourcc, 60.0, self.display_size)

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        self.recording_mode = None

    def capture_frame_if_active(self):
        if not self.recording or self.video_writer is None: return
        buffer = glReadPixels(0, 0, self.display_size[0], self.display_size[1], GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(buffer, dtype=np.uint8)
        image = image.reshape((self.display_size[1], self.display_size[0], 3))
        image = np.flipud(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.video_writer.write(image)

    def run(self):
        clock = pygame.time.Clock()
        running = True
        print("SISTEMA DE COLISÃO ATIVO.")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: self.paused = not self.paused
                    if event.key == pygame.K_r: self.cam_dist = 50; self.cam_pitch = 30; self.cam_center = [0, 0, 0]
                    if event.key == pygame.K_t: self.cam_pitch = 89; self.cam_yaw = 0
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
                self.sim_time += self.global_dt  # Atualiza tempo simulado

                # 1. Update Movimento
                for d in self.drones:
                    d.update()

                # 2. Checa Colisões
                self.check_collisions()

            # Renderização
            glClearColor(0.05, 0.05, 0.2, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.update_camera()
            self.draw_environment()

            for o in self.obstacles: o.draw()
            for d in self.drones:
                d.draw_path()
                d.draw()

            # Desenha Interface de Texto (Overlay)
            self.draw_text_overlay()

            self.capture_frame_if_active()

            rec_text = f"[● REC {self.recording_mode}]" if self.recording else ""
            status = "PAUSADO" if self.paused else "PLAY"
            pygame.display.set_caption(f"Drone Sim 3D | {status} | Time: {self.sim_time:.1f}s | {rec_text}")
            pygame.display.flip()
            clock.tick(60)

        if self.video_writer: self.video_writer.release()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    if not os.path.exists("config.json"):
        print("Gerando config para teste de colisão...")
        config_data = {
            "simulacao": {"passo_tempo_global": 0.2, "duracao_maxima": 2000},
            "ambiente": {"limite_ceu": 60, "limite_solo": 60},
            "obstaculos": [{"id": "torre_central", "posicao": [0, 0, 0], "dimensoes": {"altura": 15, "raio": 2},
                            "colisional": True, "transparencia": 40}],
            "drones": [
                # Drones configurados para colidir (mesmo raio 5, velocidades opostas)
                {"id": "kamikaze_1", "tipo_trajetoria": "orbital", "cor_rgb": [1, 0, 0], "posicao_inicial": [5, 5, 0],
                 "posicao_final": [0, 0, 0], "area_seguranca": 1.5, "raio_orbita": 5.0, "velocidade_orbital": 1.0},
                {"id": "kamikaze_2", "tipo_trajetoria": "orbital", "cor_rgb": [0, 0, 1], "posicao_inicial": [-5, 5, 0],
                 "posicao_final": [0, 0, 0], "area_seguranca": 1.5, "raio_orbita": 5.0, "velocidade_orbital": -1.0}
            ]
        }
        with open("config.json", 'w') as f: json.dump(config_data, f, indent=4)

    app = SimulationEngine("config.json")
    app.run()
