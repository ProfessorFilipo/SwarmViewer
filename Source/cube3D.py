import math


class Cube3D:
    def __init__(self, center_x, center_y, center_z, size):
        """
        Inicializa um cubo 3D com centro e tamanho definidos.
        """
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.center_z = float(center_z)
        self.size = float(size)

        # Distância do centro até a face (metade do tamanho)
        s = self.size / 2.0

        # --- CORREÇÃO 1: Inicialização Determinística dos Vértices ---
        # Face Frontal (Z Negativo)
        self.p1 = [self.center_x - s, self.center_y - s, self.center_z - s]
        self.p2 = [self.center_x + s, self.center_y - s, self.center_z - s]
        self.p3 = [self.center_x + s, self.center_y + s, self.center_z - s]
        self.p4 = [self.center_x - s, self.center_y + s, self.center_z - s]

        # Face Traseira (Z Positivo)
        self.p5 = [self.center_x - s, self.center_y - s, self.center_z + s]
        self.p6 = [self.center_x + s, self.center_y - s, self.center_z + s]
        self.p7 = [self.center_x + s, self.center_y + s, self.center_z + s]
        self.p8 = [self.center_x - s, self.center_y + s, self.center_z + s]

        # Lista de referência para iterar
        self.vertices = [self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8]

        # --- CORREÇÃO 2: Atribuição Explícita de Cores ---
        # Lista de 8 cores (RGB), uma para cada vértice
        self.colors = [(10,10,10), (10,10,10), (10,10,10), (10,10,10), (10,10,10), (10,10,10), (10,10,10), (10,10,10)]

    def rotate(self, angle, axis):
        """
        Aplica rotação aos vértices do cubo em torno do seu centro.
        """
        rad = angle
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        cx, cy, cz = self.center_x, self.center_y, self.center_z

        for v in self.vertices:
            # 1. Translação para a origem
            x = v - cx
            y = v[1] - cy
            z = v[2] - cz

            # 2. Aplicação da Matriz de Rotação (Sistema Mão Direita)
            if axis == 'x':
                # Rotação X (Roll)
                y_new = y * cos_a - z * sin_a
                z_new = y * sin_a + z * cos_a
                v[1] = y_new + cy
                v[2] = z_new + cz

            elif axis == 'y':
                # Rotação Y (Yaw)
                x_new = x * cos_a + z * sin_a
                z_new = -x * sin_a + z * cos_a
                v = x_new + cx
                v[2] = z_new + cz

            elif axis == 'z':
                # Rotação Z (Pitch)
                x_new = x * cos_a - y * sin_a
                y_new = x * sin_a + y * cos_a
                v = x_new + cx
                v[1] = y_new + cy


if __name__ == "__main__":
    cubo = Cube3D(400, 300, 0, 100)
    print("Cubo criado com sucesso!")
    print(f"Vértice P1: {cubo.p1}")
    print(f"Cores: {len(cubo.colors)}")
