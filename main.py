import csv
import random
import numpy as np
from reversi_nb import Reversi, NaiveBayes, select_move


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
