"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if board == initial_state():
        return X

    nX = 0
    nO = 0
    for row in board:
        nX += row.count(X)
        nO += row.count(O)

    if nX > nO:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    le_ac = set()
    for row in range(3):
        for cell in range(3):
            if board[row][cell] == EMPTY:
                le_ac.add((row, cell))

    return le_ac


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action[0] not in range(0, 3) or action[1] not in range(0, 3) or board[action[0]][action[1]] is not EMPTY:
        raise Exception("Invalid move")

    nb = copy.deepcopy(board)
    nb[action[0]][action[1]] = player(board)
    return nb


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for score in [X, O]:
        
        for row in range(0, 3):
            if all(board[row][col] == score for col in range(0, 3)):
                return score
        
        for col in range(0, 3):
            if all(board[row][col] == score for row in range(0, 3)):
                return score

        diagonals = [[(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]
        for diagonal in diagonals:
            if all(board[row][col] == score for (row, col) in diagonal):
                return score

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    
    if winner(board) is not None:
        return True
    
    all_moves = [cell for row in board for cell in row]
    if not any(move == EMPTY for move in all_moves):
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    if player(board) == X:
        best_q = -math.inf
        for move in actions(board):
            max_q = min_val(result(board, move))
            if max_q > best_q:
                best_q = max_q
                best_move = move
    
    elif player(board) == O:
        best_q = math.inf
        for move in actions(board):
            min_q = max_val(result(board, move)) 
            if min_q < best_q:
                best_q = min_q
                best_move = move
    return best_move 


def min_val(board):
    """
    Returns the minimum utility of the current board.
    """

    if terminal(board):
        return utility(board)
    
    q = math.inf
    for move in actions(board):
        q = min(q, max_val(result(board, move)))
    return q


def max_val(board):
    """
    Returns the maximum utility of the current board.
    """

    if terminal(board):
        return utility(board)

    q = -math.inf
    for move in actions(board):
        q = max(q, min_val(result(board, move)))
    return q
