import csv
import random
import numpy as np
from sklearn.naive_bayes import GaussianNB
from itertools import product


class Reversi:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3][3] = 1
        self.board[4][4] = 1
        self.board[3][4] = -1
        self.board[4][3] = -1
        self.turn = 1

    def get_valid_moves(self):
        valid_moves = []
        for i, j in product(range(8), range(8)):
            if self.board[i][j] == 0:
                for di, dj in product([-1, 0, 1], [-1, 0, 1]):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if (
                        ni >= 0
                        and ni < 8
                        and nj >= 0
                        and nj < 8
                        and self.board[ni][nj] == -self.turn
                    ):
                        ni += di
                        nj += dj
                        while (
                            ni >= 0
                            and ni < 8
                            and nj >= 0
                            and nj < 8
                            and self.board[ni][nj] == -self.turn
                        ):
                            ni += di
                            nj += dj
                        if (
                            ni >= 0
                            and ni < 8
                            and nj >= 0
                            and nj < 8
                            and self.board[ni][nj] == self.turn
                        ):
                            valid_moves.append((i, j))
                            break
        return valid_moves

    def make_move(self, move):
        i, j = move
        self.board[i][j] = self.turn
        for di, dj in product([-1, 0, 1], [-1, 0, 1]):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if (
                ni >= 0
                and ni < 8
                and nj >= 0
                and nj < 8
                and self.board[ni][nj] == -self.turn
            ):
                ni += di
                nj += dj
                while (
                    ni >= 0
                    and ni < 8
                    and nj >= 0
                    and nj < 8
                    and self.board[ni][nj] == -self.turn
                ):
                    ni += di
                    nj += dj
                if (
                    ni >= 0
                    and ni < 8
                    and nj >= 0
                    and nj < 8
                    and self.board[ni][nj] == self.turn
                ):
                    ni, nj = i + di, j + dj
                    while self.board[ni][nj] == -self.turn:
                        self.board[ni][nj] = self.turn
                        ni += di
                        nj += dj
        self.turn = -self.turn

    def get_score(self):
        return np.sum(self.board)

    def check_game_over(self):
        count_zero = 64 - np.count_nonzero(self.board)
        return count_zero == 0 or not self.get_valid_moves()


class NaiveBayes:
    def __init__(self):
        # self.clf = MultinomialNB()
        self.clf = GaussianNB()

    def train(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


def evaluate(rs: Reversi, nb: NaiveBayes):
    return nb.predict_proba([rs.board.flatten()])[0][0]


def negamax(rs: Reversi, nb: NaiveBayes, depth, alpha, beta):
    # if depth == 0 or rs.is_game_over():
    if depth == 0:
        return rs.turn * evaluate(rs, nb)
    score = -np.inf
    for move in rs.get_valid_moves():
        next_rs = Reversi()
        next_rs.board = np.copy(rs.board)
        next_rs.turn = rs.turn
        next_rs.make_move(move)
        score = max(score, -negamax(next_rs, nb, depth - 1, -beta, -alpha))
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return score


def select_move(rs: Reversi, nb: NaiveBayes, depth):
    best_move = None
    best_score = -np.inf
    for move in rs.get_valid_moves():
        next_rs = Reversi()
        next_rs.board = np.copy(rs.board)
        next_rs.turn = rs.turn
        next_rs.make_move(move)
        score = -negamax(next_rs, nb, depth - 1, -np.inf, np.inf)
        if score > best_score:
            best_move = move
            best_score = score
    return best_move


# create training data
X_train = []
y_train = []

with open("train_data.csv", newline="") as file:
    reader = csv.reader(file, delimiter=",")
    for row in reader:
        X_train.append(list(map(int, row[:-1])))
        y_train.append(int(row[-1]))


# train Naive Bayes model
nb = NaiveBayes()
nb.train(X_train, y_train)


# play a game using Negamax and Naive Bayes
rs = Reversi()

while True:
    if rs.turn == 1:
        move = select_move(rs, nb, 4)
        if not move:
            break
    else:
        valid_moves = rs.get_valid_moves()
        if not valid_moves:
            break
        # move = valid_moves[np.random.randint(len(valid_moves))]
        move = random.choice(valid_moves)
    # print("Player", rs.turn, "moves to", move)
    rs.make_move(move)
    # print(rs.board)

# print(rs.board)
print("Final score:", rs.get_score())
