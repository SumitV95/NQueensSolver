"""
@author: Sumeet Vadhavkar
"""



import numpy
from GameBoard import GameBoard
import math
import random
import sys
import threading
import matplotlib.pyplot as plt


class MyThread(threading.Thread):

    def __init__(self, boards, populationSize, probabilityDistribution, arr, str, end, maxHeuristic):
        threading.Thread.__init__(self)
        self.boards = boards
        self.populationSize = populationSize
        self.probabilityDistribution = probabilityDistribution
        self.arr = arr
        self.str = str
        self.end = end
        self.sumOfBoards = 0
        self.newBoards = []
        self.solutionFound = False
        self.maxHeuristic = maxHeuristic

    def run(self) -> None:
        parentArr = []
        for i, parent in enumerate(self.arr[self.str:self.end]):
            parentArr.append(parent)
            if i % 2 != 0 or i == self.end - 1:
                if self.end - self.str % 2 != 0 and i == self.end - 1:
                    child1, child2 = generateBoards(self.arr[self.end - 1], self.arr[self.end - 2], boardSize)
                else:
                    child1, child2 = generateBoards(parentArr[0], parentArr[1], boardSize)
                parentArr = []
                num = random.randint(0, 100)
                if num < 5:
                    mutation(child1)

                num = random.randint(0, 100)
                if num < 5:
                    mutation(child2)
                child1.heuristic = int((totalPossible - child1.numberOfAttackingPairs()) ** 2)
                child2.heuristic = int((totalPossible - child2.numberOfAttackingPairs()) ** 2)
                if child1.heuristic > self.maxHeuristic:
                    self.maxHeuristic = child1.heuristic
                if child2.heuristic > maxHeuristic:
                    self.maxHeuristic = child2.heuristic
                if child1.heuristic == totalPossible ** 2:
                    print("this is the solution:")

                    print(child1.array)
                    self.solutionFound = True
                    break
                if child2.heuristic == totalPossible ** 2:
                    print("this is the solution:")

                    print(child2.array)
                    self.solutionFound = True
                    break

                self.newBoards.append(child1)

                self.newBoards.append(child2)

                self.sumOfBoards += child1.heuristic
                self.sumOfBoards += child2.heuristic


def generateBoards(parent1, parent2, boardSize):
    """
    This function creates two new children from the parent boards provided
    :param parent1: First parent
    :param parent2: second parent
    :param boardSize: size of the board
    :return: the children boards
    """
    board1 = GameBoard(boardSize)
    board2 = GameBoard(boardSize)
    slice = boardSize // 2
    board1.array = numpy.concatenate((parent1.array[:slice], parent2.array[slice:]))
    board2.array = numpy.concatenate((parent2.array[:slice], parent1.array[slice:]))
    board1.queens = parent1.queens[:slice] + parent2.queens[slice:]
    board2.queens = parent2.queens[:slice] + parent1.queens[slice:]
    return board1, board2


def combinationCalculator(n, r):
    """
    calculates the combinations
    """

    return math.factorial(n) / (math.factorial(n - r) * math.factorial(r))


def generateFirstPopulation(populationSize):
    """

    :param populationSize: number of boards to generate
    :return: boards and the probability associated with those boards
    """
    sumOfBoards = 0
    boards = []
    for i in range(populationSize):
        board = GameBoard(boardSize)
        board.initializeGameBoard()
        numberOfAttacks = board.numberOfAttackingPairs()
        boards.append(board)
        board.heuristic = (totalPossible - numberOfAttacks) ** 2
        sumOfBoards += (totalPossible - numberOfAttacks) ** 2
    probabilityDistribution = []
    for i in range(len(boards)):
        probabilityDistribution.append(boards[i].heuristic / sumOfBoards)
    return boards, probabilityDistribution


def mutation(child1):
    """

    :param child1: child board
    :return:
    """
    # print("Before Mutation:")
    # print(child1.array)
    n = child1.N
    column = random.randint(0, n - 1)
    currentLoc = child1.queens[column]
    move = random.randint(0, n - 1)
    while move == currentLoc[1]:
        move = random.randint(0, n - 1)
    child1.queens[column] = (column, move)
    child1.array[column][currentLoc[1]] = 0
    child1.array[column][move] = 1

    # print("mutation occurred")
    # print(child1.array)


def runGeneticAlgorithm(epoch, boards, populationSize, probabilityDistribution, maxHeuristic):
    """

    :param epoch: max number of epochs
    :param boards: list of boards
    :param populationSize: size of the population
    :param probabilityDistribution: probabilities associated with the boards
    :param maxHeuristic: current maximum value of the heuristic function
    :return:
    """
    j = 1
    yPlot = []
    xPlot = []
    while epoch > 0:

        newBoards = []

        newSumOfBoards = 0

        arr = numpy.random.choice(boards, populationSize, True, probabilityDistribution)
        parentArr = []
        for i, parent in enumerate(arr):
            parentArr.append(parent)
            if i % 2 != 0:
                child1, child2 = generateBoards(parentArr[0], parentArr[1], boardSize)
                parentArr = []
                num = random.randint(0, 100)
                if num < 5:
                    mutation(child1)

                num = random.randint(0, 100)
                if num < 5:
                    mutation(child2)
                child1.heuristic = int((totalPossible - child1.numberOfAttackingPairs()) ** 2)
                child2.heuristic = int((totalPossible - child2.numberOfAttackingPairs()) ** 2)
                if child1.heuristic > maxHeuristic:
                    maxHeuristic = child1.heuristic
                if child2.heuristic > maxHeuristic:
                    maxHeuristic = child2.heuristic
                if child1.heuristic == totalPossible ** 2:
                    print("this is the solution:")
                    print(epoch)
                    print(child1.array)
                    epoch = 0
                    break
                if child2.heuristic == totalPossible ** 2:
                    print("this is the solution:")
                    print(epoch)
                    print(child2.array)
                    epoch = 0
                    break

                newBoards.append(child1)

                newBoards.append(child2)

                newSumOfBoards += child1.heuristic
                newSumOfBoards += child2.heuristic

        sumOfBoards = newSumOfBoards
        boards = newBoards
        for i in range(len(boards)):
            probabilityDistribution[i] = boards[i].heuristic / sumOfBoards

        print(maxHeuristic)
        print(epoch)
        epoch -= 1
        yPlot.append(sumOfBoards / len(boards))
        xPlot.append(j)
        j += 1
    print(xPlot)
    print(yPlot)
    plt.plot(xPlot, yPlot)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Fitness Value of Generation")
    plt.show()

    print(maxHeuristic)


def runMultiThreadedAlgorithm(epoch, boards, populationSize, probabilityDistribution, maxHeuristic):
    solutionFound = False
    j = 0
    xPlot = []
    yPlot = []
    while epoch > 0:

        newSumOfBoards = 0

        arr = numpy.random.choice(boards, populationSize, True, probabilityDistribution)
        size = populationSize // 3
        start = 0
        end = start + size
        threads = []
        for i in range(3):
            threads.append(
                MyThread(boards, populationSize, probabilityDistribution, arr, start, end, maxHeuristic))

            threads[i].start()
            start = end
            if i != 1:
                end = start + size
            else:
                end = len(arr)
        for thread in threads:
            thread.join()
        for thread in threads:
            if thread.solutionFound is True:
                solutionFound = True
                break
            newSumOfBoards += thread.sumOfBoards

        if solutionFound is True:
            break
        i = 0
        for thread in threads:
            for board in thread.newBoards:
                boards[i] = board
                probabilityDistribution[i] = boards[i].heuristic / newSumOfBoards
                i += 1

        yPlot.append(newSumOfBoards / len(boards))
        xPlot.append(j)
        epoch -= 1
        j += 1
        print(epoch)
    plt.plot(xPlot, yPlot)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Fitness Value of Generation")
    plt.show()


epoch = sys.argv[1]
epoch = int(epoch)
boardSize = sys.argv[2]
boardSize = int(boardSize)
populationSize = sys.argv[3]
populationSize = int(populationSize)
totalPossible = combinationCalculator(boardSize, 2)

print(totalPossible)
boards, probabilityDistribution = generateFirstPopulation(populationSize)
print(boards)
maxHeuristic = 0
if populationSize <= 5000:
    runGeneticAlgorithm(epoch, boards, populationSize, probabilityDistribution, maxHeuristic)
else:
    runMultiThreadedAlgorithm(epoch, boards, populationSize, probabilityDistribution, maxHeuristic)
