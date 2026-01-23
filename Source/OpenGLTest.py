
############################################################################
####                      O P E N G L     T E S T                       ####
############################################################################
#### Show a yellow cube (wireframe only) spinning in a blue background) ####
############################################################################

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys


def main():
    print("1. Inicializando Pygame...")
    pygame.init()

    # Configurações para evitar tela preta em drivers modernos
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

    display = (800, 600)
    print(f"2. Criando Janela {display}...")
    try:
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        print("   -> Janela criada com sucesso.")
    except Exception as e:
        print(f"   -> ERRO ao criar janela: {e}")
        return

    # Configuração da Câmera
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)  # Move a câmera 5 unidades para trás

    # Verifica se OpenGL inicializou
    try:
        version = glGetString(GL_VERSION)
        print(f"3. Versão OpenGL detectada: {version}")
    except:
        print("3. ERRO: Não foi possível ler a versão do OpenGL.")

    print("4. Iniciando Loop...")

    rotation = 0
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Rotação automática
        rotation += 1

        # Limpa a tela com AZUL (R, G, B, Alpha)
        # Se você ver azul, o OpenGL está funcionando.
        glClearColor(0.0, 0.0, 0.5, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Desenha um Cubo
        glPushMatrix()
        glRotatef(rotation, 1, 1, 0)  # Gira nos eixos X e Y

        glBegin(GL_LINES)
        glColor3f(1, 1, 0)  # Amarelo
        # Cubo simples (apenas algumas linhas para teste)
        vertices = [
            (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
            (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

        glPopMatrix()

        pygame.display.flip()
        clock.tick(60)

    print("Encerrando.")
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
