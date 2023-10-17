import base64
import datetime
import math
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

class SimpleGA:

    def __init__(self, config):
        # Максимальное количество хромосом в популяции
        self.max_population = config['max_population']
        # Максимальное количество эпох
        self.max_epochs = config['max_epochs']
        # Актуальная эпоха
        self.current_epoch = 0
        # Вероятность кроссовера
        self.crossover_chance = config['crossover_chance']
        # Вероятность мутации
        self.mutation_chance = config['mutation_chance']
        # Нижняя граница
        self.lower_bound = float(config['lower_bound'])
        # Верхняя граница
        self.upper_bound = config['upper_bound']
        # Количество точек, на которые делим отрезок
        self.chromosome_length = 15
        # Коды для этих точек
        self.codes = [format(x, '015b') for x in range(0, 2**self.chromosome_length)]
        # Как из кода хромосомы получить точку на отрезке
        self.from_binary_to_number = lambda x: round(self.lower_bound + int(x, 2) * ((self.upper_bound - self.lower_bound)
                                                                       / (2**self.chromosome_length - 1)), 3)
        #Функция, с которой работаем
        self.function = lambda x: (math.sin(2*x))/x**2
        # Актуальная популяция хромосом
        self.population = [self.codes[random.randint(0, 2**self.chromosome_length-1)] for x in range(0, self.max_population)]
        # Потомки актуальной популяции
        self.children = []
        # Лучшее решение актуальной популяции
        self.current_best_solution = -sys.maxsize - 1

    def selection(self):
        reselected_population = []
        avg = np.mean([self.function(self.from_binary_to_number(x)) for x in self.population])
        fitness_function = [self.function(self.from_binary_to_number(x))/avg for x in self.population]
        for i in range(0, self.max_population):
           for ff in range(0, int(abs(fitness_function[i]))):
               reselected_population.append(self.population[i])
           if random.uniform(0, 1) <= int(abs(fitness_function[i]) % 1 * 1000):
               reselected_population.append(self.population[i])
        self.population = reselected_population

    def crossover(self):
        for i in range(0, len(self.population)):
            if random.uniform(0, 1) <= self.crossover_chance:
                chrom_a = self.population[i]

                chrom_b = self.population[random.randint(0, len(self.population)-1)]

                k = random.randint(0, self.chromosome_length-1)

                chrom_a_ = chrom_a[:k] + chrom_b[k:]
                chrom_b_ = chrom_b[:k] + chrom_a[k:]

                self.children.append(chrom_a_)
                self.children.append(chrom_b_)

    def mutation(self):
        for i in range(0, len(self.children)):
            if round(random.uniform(0, 1), 3) <= self.mutation_chance:
                a_place = random.randint(0, len(self.children)-1)
                chrom_a = self.children[a_place]
                k = random.randint(0, self.chromosome_length-1)
                temp = list(chrom_a)
                if chrom_a[k] == '0':
                    temp[k] = '1'
                else:
                    temp[k] = '0'
                chrom_a_ = ''.join(temp)
                self.children[a_place] = chrom_a_

    def sort(self):
        return lambda x: self.function(self.from_binary_to_number(x))

    def reduction(self):
        self.population = self.population + self.children
        self.population.sort(key=self.sort(), reverse=True)
        self.population = self.population[:self.max_population]

    def plot_graph(self):
        x = np.arange(self.lower_bound, self.upper_bound + 0.1, 0.1)
        y = [self.function(i) for i in x]
        x_population = [self.from_binary_to_number(x) for x in self.population]
        y_population = [self.function(x) for x in x_population]
        plt.plot(x,y)
        plt.scatter(x_population, y_population, alpha=0.5, c=[np.arange(self.max_population)])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'График функции для {self.current_epoch} эпохи')
        plt.grid(True)
        plt.show()

    def run(self):
        epochs_without_changes = 0
        delta = 0.001
        self.plot_graph()
        while self.current_epoch != self.max_epochs and epochs_without_changes != 10:
            self.current_epoch += 1
            self.selection()
            self.crossover()
            self.mutation()
            self.reduction()
            best_solution = max([self.function(self.from_binary_to_number(x)) for x in self.population])
            print(
f"""
++++++++++
Эпоха: {self.current_epoch}
Лучшее решение эпохи: {best_solution}
Лучшее решение за все эпохи: {self.current_best_solution}
Количество эпох без изменения результата: {epochs_without_changes}
++++++++++
"""
            )
            if round(abs(best_solution - self.current_best_solution)) <= delta:
                epochs_without_changes += 1
            elif best_solution > self.current_best_solution:
                self.current_best_solution = best_solution
                epochs_without_changes = 0
            else:
                epochs_without_changes = 0
            self.plot_graph()

def main():
    config = {
        "crossover_chance" : 0.6,
        "mutation_chance" : 0.001,
        "max_population" : 50,
        "max_epochs" : 100,
        "lower_bound" : -20,
        "upper_bound" : -3.1,
    }
    GA = SimpleGA(config)
    GA.run()

if __name__ == '__main__':
    random.seed(round(time.time()))
    main()
