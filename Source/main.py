##################################################################
####                 S W A R M    V I E W E R                 ####
##################################################################
#### Visualizador 3D de Enxames de Drones com PyGame          ####
#### v2.5 - Correção da Classe Environment e Inicialização    ####
##################################################################

import pygame
import numpy as np
import math

# --- CONSTANTES DE CONFIGURAÇÃO ---
WIDTH, HEIGHT = 1280, 720
FPS = 60
NUM_DRONES = 250
VISUAL_RANGE = 45.0
PROTECTED_RANGE = 9.0
FACTOR_COHESION = 0.0008
FACTOR_ALIGNMENT = 0.05
FACTOR_SEPARATION = 0.06
MAX_SPEED = 4.0
MIN_SPEED = 2.0
DRONE_SIZE = 6

# Dimensões do Mundo Virtual
WORLD_SIZE = 300
FLOOR_Y = 200
CEILING_Y = -200

# Cores
COLOR_BG = (15, 15, 30)
COLOR_DRONE = (0, 255, 255)
COLOR_GLOW = (0, 100, 255)
COLOR_GRID_FLOOR = (0, 100, 80)
COLOR_GRID_SKY = (50, 50, 80)


class Camera:
    def __init__(self, width, height):
        self.position = np.array([0.0, -100.0, -500.0])
        self.fov = 850.0
        self.width = width
        self.height = height
        self.angle_x = 0.2
        self.angle_y = 0.0

    def update(self):
        keys = pygame.key.get_pressed()
        if keys: self.angle_y -= 0.03
        if keys: self.angle_y += 0.03
        if keys[pygame.K_UP]: self.angle_x -= 0.03
        if keys: self.angle_x += 0.03

        speed = 5.0
        if keys[pygame.K_w]: self.position[1] += speed
        if keys[pygame.K_s]: self.position[1] -= speed
        if keys[pygame.K_a]: self.position -= speed
        if keys[pygame.K_d]: self.position += speed
        if keys[pygame.K_q]: self.position[2] -= speed
        if keys[pygame.K_e]: self.position[2] += speed

    def project(self, points_3d):
        """ Pipeline 3D -> 2D Otimizado """
        cy, sy = math.cos(-self.angle_y), math.sin(-self.angle_y)
        cx, sx = math.cos(-self.angle_x), math.sin(-self.angle_x)

        # Matriz de Rotação Y (Yaw)
        rot_y = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])

        # Matriz de Rotação X (Pitch)
        rot_x = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])

        rotation_matrix = rot_x @ rot_y

        points_rel = points_3d - self.position
        # Transposta para multiplicar (3,3) @ (3, N)
        points_cam = (rotation_matrix @ points_rel.T).T

        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]

        z_safe = np.maximum(z, 1.0)
        factor = self.fov / z_safe

        x_proj = x * factor + self.width / 2
        y_proj = y * factor + self.height / 2

        return np.column_stack((x_proj, y_proj)), z


class Environment:
    def __init__(self):
        # CORREÇÃO: Inicialização correta como listas vazias
        self.lines = []
        self.colors = []

        grid_size = 600
        step = 100

        # Gera linhas do Grid
        for i in range(-grid_size, grid_size + 1, step):
            # --- Linhas paralelas ao eixo Z (Fundo <-> Frente) ---

            # Chão (Y = FLOOR_Y)
            p1 = np.array(i, FLOOR_Y, -grid_size)
            p2 = np.array(i, FLOOR_Y, grid_size)
            self.lines.append([p1, p2])
            self.colors.append(COLOR_GRID_FLOOR)

            # Teto (Y = CEILING_Y)
            p3 = np.array()
            p4 = np.array()
            self.lines.append([p3, p4])
            self.colors.append(COLOR_GRID_SKY)

            # --- Linhas paralelas ao eixo X (Esquerda <-> Direita) ---

            # Chão
            p5 = np.array(-grid_size, FLOOR_Y, i)
            p6 = np.array(grid_size, FLOOR_Y, i)
            self.lines.append([p5, p6])
            self.colors.append(COLOR_GRID_FLOOR)

            # Teto
            p7 = np.array()
            p8 = np.array()
            self.lines.append([p7, p8])
            self.colors.append(COLOR_GRID_SKY)

        # Converte lista plana de pontos para Numpy Array para projeção rápida
        self.flat_points_list = []
        for start, end in self.lines:
            self.flat_points_list.append(start)
            self.flat_points_list.append(end)
        self.flat_points = np.array(self.flat_points_list)

    def draw(self, screen, camera):
        # Projeta todos os pontos de uma vez
        points_2d, depths = camera.project(self.flat_points)

        # Desenha as linhas recuperando os pares
        # Cada linha consome 2 pontos do array projetado
        for i in range(0, len(points_2d), 2):
            p1 = points_2d[i]
            p2 = points_2d[i + 1]
            z1 = depths[i]
            z2 = depths[i + 1]

            # Só desenha se ambos os pontos estiverem na frente da câmera (z > 1)
            if z1 > 1.0 and z2 > 1.0:
                color_index = i // 2
                color = self.colors[color_index]
                pygame.draw.aaline(screen, color, p1, p2)


class Swarm:
    def __init__(self, count):
        self.positions = (np.random.rand(count, 3) - 0.5) * WORLD_SIZE
        self.velocities = (np.random.rand(count, 3) - 0.5) * MAX_SPEED
        self.count = count

    def update(self):
        # Matrizes de distância (Numpy Broadcasting)
        diff_matrix = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)

        mask_visual = dist_matrix < VISUAL_RANGE
        mask_protected = dist_matrix < PROTECTED_RANGE

        # 1. Separação
        separation = np.sum(diff_matrix * mask_protected[:, :, np.newaxis], axis=1) * FACTOR_SEPARATION

        # 2. Alinhamento
        counts = np.maximum(np.sum(mask_visual, axis=1)[:, np.newaxis], 1)
        avg_vel = np.sum(self.velocities[np.newaxis, :, :] * mask_visual[:, :, np.newaxis], axis=1) / counts
        alignment = (avg_vel - self.velocities) * FACTOR_ALIGNMENT

        # 3. Coesão
        avg_pos = np.sum(self.positions[np.newaxis, :, :] * mask_visual[:, :, np.newaxis], axis=1) / counts
        cohesion = (avg_pos - self.positions) * FACTOR_COHESION

        # Zera forças se isolado
        has_neighbors = np.sum(mask_visual, axis=1) > 0
        alignment[~has_neighbors] = 0
        cohesion[~has_neighbors] = 0

        # Força suave de retorno ao centro
        return_force = -self.positions * 0.0005

        # Colisão com Chão/Teto (Reforço)
        floor_mask = self.positions[:, 1] > FLOOR_Y - 10
        self.velocities[floor_mask, 1] -= 0.5
        ceil_mask = self.positions[:, 1] < CEILING_Y + 10
        self.velocities[ceil_mask, 1] += 0.5

        # Aplica Forças
        self.velocities += separation + alignment + cohesion + return_force

        # Limita Velocidade
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
        pygame.display.set_caption("Swarm Viewer 2.5 - Fixed Environment")
        self.clock = pygame.time.Clock()
        self.running = True

        self.swarm = Swarm(NUM_DRONES)
        self.camera = Camera(WIDTH, HEIGHT)
        self.environment = Environment()

        # Textura de Glow
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

            # Lógica
            self.swarm.update()
            self.camera.update()

            # Desenho
            self.screen.fill(COLOR_BG)

            # 1. Ambiente (Chão e Teto)
            self.environment.draw(self.screen, self.camera)

            # 2. Drones
            points_2d, depths_z = self.camera.project(self.swarm.positions)
            sorted_indices = np.argsort(depths_z)[::-1]

            for i in sorted_indices:
                z = depths_z[i]
                if z < 1.0: continue

                x, y = points_2d[i]

                # Culling (Só desenha se estiver na tela)
                if -50 <= x <= WIDTH + 50 and -50 <= y <= HEIGHT + 50:
                    scale = min(max(400 / z, 0.4), 6.0)
                    current_size = int(DRONE_SIZE * scale)
                    brightness = min(max(600 / z, 0.2), 1.0)

                    # Cor com brilho ajustado
                    r = int(COLOR_DRONE * brightness)
                    g = int(COLOR_DRONE[2] * brightness)
                    b = int(COLOR_DRONE[1] * brightness)
                    color = (r, g, b)

                    # Shadow / Drop Line (Sombra no chão)
                    # Cria ponto no chão com mesmo X, Z do drone
                    floor_pos = np.array([self.swarm.positions[i, 0], FLOOR_Y, self.swarm.positions[i, 2]])
                    # Projeta esse único ponto (reshape para 2D array)
                    floor_2d_arr, f_z = self.camera.project(floor_pos[np.newaxis, :])

                    if f_z > 1.0 and scale > 0.5:
                        fx, fy = floor_2d_arr
                        # Linha vertical (Drone -> Chão)
                        pygame.draw.line(self.screen, (50, 50, 50), (x, y), (fx, fy), 1)
                        # Círculo da sombra
                        pygame.draw.circle(self.screen, (30, 30, 30), (int(fx), int(fy)), max(2, int(current_size / 3)))

                    # Glow
                    if scale > 0.5:
                        glow_size = int(current_size * 4)
                        scaled_glow = pygame.transform.scale(self.glow_surf, (glow_size, glow_size))
                        self.screen.blit(scaled_glow, (x - glow_size // 2, y - glow_size // 2),
                                         special_flags=pygame.BLEND_ADD)

                    # Corpo do Drone
                    pygame.draw.circle(self.screen, color, (int(x), int(y)), max(2, int(current_size / 2)))

            # Interface HUD
            ui_text = "teste"

            for idx, line in enumerate(ui_text):
                text_surf = font.render(line, True, (200, 200, 200))
                self.screen.blit(text_surf, (10, 10 + idx * 20))

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    app = SwarmVisualizer()
    app.run()
