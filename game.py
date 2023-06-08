import csv
import pygame
import numpy as np
from itertools import product
from reversi_nb import Reversi, NaiveBayes, select_move


# global variables
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)

SQUARE = 80
MARGIN = 2
RADIUS = SQUARE / 2 - SQUARE / 10

WINDOW_SIZE = [SQUARE * 8 + MARGIN * 9, SQUARE * 8 + MARGIN * 9]


def load_data(path):
    X_train = []
    y_train = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            X_train.append(list(map(int, row[:-1])))
            y_train.append(int(row[-1]))
    return X_train, y_train


def get_position(pos):
    x, y = pos
    col = x // (MARGIN + SQUARE)
    row = y // (MARGIN + SQUARE)
    return row, col


def draw_board(screen, rs: Reversi):
    for x, y in product(range(8), range(8)):
        # checkerboard position
        nx = (MARGIN + SQUARE) * x + MARGIN
        ny = (MARGIN + SQUARE) * y + MARGIN

        # piece center
        mx = nx + SQUARE / 2
        my = ny + SQUARE / 2

        # draw checkerboard
        pygame.draw.rect(screen, GREEN, (nx, ny, SQUARE, SQUARE))

        # draw piece
        if rs.board[y][x] == 1:
            pygame.draw.circle(screen, BLACK, (mx, my), RADIUS)
        if rs.board[y][x] == -1:
            pygame.draw.circle(screen, WHITE, (mx, my), RADIUS)

        # draw valid move
        for r, c in rs.get_valid_moves():
            # checkerboard position
            px = (MARGIN + SQUARE) * c + MARGIN
            py = (MARGIN + SQUARE) * r + MARGIN

            # start, end position
            spos = (px + 3 * SQUARE / 8, py + SQUARE / 2)
            epos = (px + 5 * SQUARE / 8, py + SQUARE / 2)

            # draw line
            if rs.turn == 1:
                pygame.draw.line(screen, BLACK, spos, epos, 2)

    pygame.display.update()


def print_result(rs: Reversi):
    score = rs.get_score()
    if score > 0:
        print("You win")
    elif score < 0:
        print("You lose")
    else:
        print("Tie")


def main():
    rs = Reversi()
    nb = NaiveBayes()

    X_train, y_train = load_data("train_data.csv")
    nb.train(X_train, y_train)

    # pygame initial
    pygame.init()
    pygame.display.set_caption("Reversi Game")

    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif rs.turn == 1:
                valid_moves = rs.get_valid_moves()
                if valid_moves:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        move = get_position(pos)
                        if move in valid_moves:
                            rs.make_move(move)
                else:
                    rs.turn = -rs.turn
                    done = rs.check_game_over()
            else:
                move = select_move(rs, nb, 4)
                if move:
                    rs.make_move(move)
                else:
                    rs.turn = -rs.turn
                    done = rs.check_game_over()
        draw_board(screen, rs)
        clock.tick(60)

    print_result(rs)
    pygame.quit()


if __name__ == "__main__":
    main()
