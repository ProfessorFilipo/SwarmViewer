##################################################################
####                 S W A R M    V I E W E R                 ####
##################################################################
#### part of the Beyond Visual Sight Drone Operation Project  ####
##################################################################
#### Prof. Filipo - github.com/ProfessorFilipo/SwarmViewer    ####
##################################################################

import pygame
import numpy as np
import math

# --- CONSTANTES DE CONFIGURAÇÃO ---
WIDTH, HEIGHT = 1280, 720
FPS = 60
NUM_DRONES = 250
VISUAL_RANGE = 45.0  # Raio de percepção do drone
PROTECTED_RANGE = 9.0  # Raio de colisão (separação)
FACTOR_COHESION = 0.0008
FACTOR_ALIGNMENT = 0.05
FACTOR_SEPARATION = 0.06
MAX_SPEED = 4.0
MIN_SPEED = 2.0
DRONE_SIZE = 6

# Cores
COLOR_BG = (10, 10, 25)
COLOR_DRONE = (0, 255, 255)
COLOR_GLOW = (0, 100, 255)


class Camera:
    def __init__(self, width, height):
        self.position = np.array([0.0, 0.0, -400.0])  # Câmera afastada no eixo Z
        self.fov = 850.0
        self.width = width
        self.height = height
        self.angle_x = 0.0  # Pitch
        self.angle_y = 0.0  # Yaw

    def update(self):
        # Controle de Câmera (Setas + W/S)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: self.angle_y -= 0.05
        if keys[pygame.K_RIGHT]: self.angle_y += 0.05
        if keys[pygame.K_UP]: self.angle_x -= 0.05
        if keys[pygame.K_DOWN]: self.angle_x += 0.05

        # Zoom (Movimento no eixo Z local)
        if keys[pygame.K_w]: self.position[1] += 5.0
        if keys[pygame.K_s]: self.position[1] -= 5.0

    def project(self, points_3d):
        """
        Projeta pontos 3D (World Space) para 2D (Screen Space)
        """
        # 1. Matrizes de Rotação
        cy, sy = math.cos(-self.angle_y), math.sin(-self.angle_y)
        # CORREÇÃO: Matriz 3x3 completa e válida
        rot_y = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])

        cx, sx = math.cos(-self.angle_x), math.sin(-self.angle_x)
        # CORREÇÃO: Matriz 3x3 completa e válida
        rot_x = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])

        # Matriz Combinada
        rotation_matrix = rot_x @ rot_y

        # 2. Translação e Rotação
        points_rel = points_3d - self.position
        points_cam = (rotation_matrix @ points_rel.T).T

        # 3. Projeção Perspectiva
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]

        # Evita divisão por zero para objetos atrás da câmera
        z_safe = np.maximum(z, 1.0)

        factor = self.fov / z_safe
        x_proj = x * factor + self.width / 2
        y_proj = y * factor + self.height / 2

        return np.column_stack((x_proj, y_proj)), z


class Swarm:
    def __init__(self, count):
        # Inicia em um cubo aleatório
        self.positions = (np.random.rand(count, 3) - 0.5) * 300
        self.velocities = (np.random.rand(count, 3) - 0.5) * MAX_SPEED
        self.count = count

    def update(self):
        """
        Lógica Boids Vetorizada (Numpy)
        """
        # Matriz de Diferenças (N x N x 3)
        diff_matrix = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        # Matriz de Distâncias (N x N)
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)

        np.fill_diagonal(dist_matrix, np.inf)

        mask_visual = dist_matrix < VISUAL_RANGE
        mask_protected = dist_matrix < PROTECTED_RANGE

        # 1. Separação
        separation = np.sum(diff_matrix * mask_protected[:, :, np.newaxis], axis=1) * FACTOR_SEPARATION

        # 2. Alinhamento
        counts = np.sum(mask_visual, axis=1)[:, np.newaxis]
        counts = np.maximum(counts, 1)

        avg_vel = np.sum(self.velocities[np.newaxis, :, :] * mask_visual[:, :, np.newaxis], axis=1)
        avg_vel = avg_vel / counts
        alignment = (avg_vel - self.velocities) * FACTOR_ALIGNMENT

        # 3. Coesão
        avg_pos = np.sum(self.positions[np.newaxis, :, :] * mask_visual[:, :, np.newaxis], axis=1)
        avg_pos = avg_pos / counts
        cohesion = (avg_pos - self.positions) * FACTOR_COHESION

        # Se não tem vizinhos, ignora alinhamento/coesão
        has_neighbors = np.sum(mask_visual, axis=1) > 0
        alignment[~has_neighbors] = 0
        cohesion[~has_neighbors] = 0

        # Força de retorno (para não fugirem infinitamente)
        return_force = -self.positions * 0.0002

        # Aplica e limita
        self.velocities += separation + alignment + cohesion + return_force

        speeds = np.linalg.norm(self.velocities, axis=1)
        speeds = np.maximum(speeds, 0.0001)

        scale_factor = np.where(speeds > MAX_SPEED, MAX_SPEED / speeds, 1.0)
        scale_factor = np.where(speeds < MIN_SPEED, MIN_SPEED / speeds, scale_factor)

        self.velocities *= scale_factor[:, np.newaxis]
        self.positions += self.velocities


class SwarmVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Swarm Intelligence - Digital Twin Prototype")
        self.clock = pygame.time.Clock()
        self.running = True

        self.swarm = Swarm(NUM_DRONES)
        self.camera = Camera(WIDTH, HEIGHT)

        # Sprite de "Glow" pré-renderizado
        self.glow_surf = pygame.Surface((DRONE_SIZE * 6, DRONE_SIZE * 6), pygame.SRCALPHA)
        for r in range(DRONE_SIZE * 3, 0, -2):
            alpha = int((1 - (r / (DRONE_SIZE * 3))) * 40)
            pygame.draw.circle(self.glow_surf, (*COLOR_GLOW, alpha), (DRONE_SIZE * 3, DRONE_SIZE * 3), r)

    def run(self):
        font = pygame.font.SysFont("consolas", 14)

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: self.running = False

            # Atualiza Física
            self.swarm.update()
            self.camera.update()

            # Limpa Tela
            self.screen.fill(COLOR_BG)

            # Projeta 3D -> 2D
            points_2d, depths_z = self.camera.project(self.swarm.positions)

            # Ordena do fundo para frente (Painter's Algorithm)
            sorted_indices = np.argsort(depths_z)[::-1]

            for i in sorted_indices:
                z = depths_z[i]
                if z < 1.0: continue  # Clipping (atrás da câmera)

                x, y = points_2d[i]

                # Culling (se estiver fora da tela)
                if -50 <= x <= WIDTH + 50 and -50 <= y <= HEIGHT + 50:
                    # Perspectiva
                    scale = min(max(400 / z, 0.4), 6.0)
                    current_size = int(DRONE_SIZE * scale)

                    # Brilho baseado na distância
                    brightness = min(max(600 / z, 0.2), 1.0)

                    # CORREÇÃO DE COR: Indexação correta da tupla
                    r = int(COLOR_DRONE[0] * brightness)
                    g = int(COLOR_DRONE[1] * brightness)
                    b = int(COLOR_DRONE[2] * brightness)
                    color = (r, g, b)

                    # Desenha Glow (Aditivo)
                    if scale > 0.5:
                        glow_size = int(current_size * 4)
                        scaled_glow = pygame.transform.scale(self.glow_surf, (glow_size, glow_size))
                        # Centraliza o glow no drone
                        self.screen.blit(scaled_glow, (x - glow_size // 2, y - glow_size // 2),
                                         special_flags=pygame.BLEND_ADD)

                    # Desenha Drone
                    pygame.draw.circle(self.screen, color, (int(x), int(y)), max(2, int(current_size / 2)))

            # HUD
            ui_text = [
                f"FPS: {int(self.clock.get_fps())}",
                f"Drones: {NUM_DRONES}",
                "Controls: Arrows (Rotate) | W/S (Zoom)"
            ]

            for idx, line in enumerate(ui_text):
                text_surf = font.render(line, True, (200, 200, 200))
                self.screen.blit(text_surf, (10, 10 + idx * 20))

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    app = SwarmVisualizer()
    app.run()