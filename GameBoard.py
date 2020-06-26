import numpy
import random


def calculateSlope(queen1, queen2):
    """

    :param queen1: location of queen1
    :param queen2: location of queen2
    :return: the slope between them
    """

    if queen2[0] - queen1[0] == 0:
        return float("inf")
    y = (queen2[1] - queen1[1])
    x = (queen2[0] - queen1[0])

    return y/x


class GameBoard:
    def __init__(self, N):
        self.N = N
        self.array = numpy.zeros((self.N, self.N))
        self.queens = []
        self.range = None
        self.heuristic = None

    def initializeGameBoard(self):
        """
        this function initializes the board
        :return:
        """

        for i in range(self.N):
            queen = random.randint(0, self.N - 1)
            self.array[i][queen] = 1
            self.queens.append((i, queen))
        #print(self.array)
        #print(self.queens)

    def numberOfAttackingPairs(self):
        """

        :return:calculates the number of attacking pairs for the board
        """

        pairs = 0
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):

                slope = calculateSlope(self.queens[i], self.queens[j])

                if slope == 0 or slope == float("inf") or slope == 1 or slope == -1:

                    pairs += 1
        return pairs
