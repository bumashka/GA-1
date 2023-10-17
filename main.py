import base64
import math
import random
import matplotlib.pyplot as plt
import numpy as np

class SimpleGA:

    def __init__(self, config):
        # Максимальное количество хромосом в популяции
        self.max_population = config['max_population']
        # Эпоха
        self.max_epochs = config['max_epochs']
        self.crossover_chance = config['crossover_chance']
        self.mutation_chance = config['mutation_chance']
        self.current_epoch = 0
        # Нижняя граница
        self.lower_bound = float(config['lower_bound'])
        self.delta = 0.001
        self.current_best_solution = float(self.lower_bound)
        # Верхняя граница
        self.upper_bound = config['upper_bound']
        # Количество точек, на которые делим отрезок
        self.chromosome_length = 15
        self.max_number_of_slices = 2**self.chromosome_length
        # Точки, на которые делим отрезок
        self.points = np.linspace(self.lower_bound, self.upper_bound, self.max_number_of_slices)
        # Коды для этих точек
        self.codes = [format(x, '015b') for x in range(0, len(self.points))]
        # Как из кода хромосомы получить точку на отрезке
        self.from_binary_to_number = lambda x: round(self.lower_bound + int(x, 2) * ((self.upper_bound - self.lower_bound)
                                                                       / (self.max_number_of_slices - 1)), 3)
        self.function = lambda x: (math.sin(2*x))/x**2
        # Актуальная популяция хромосом
        self.population = [self.codes[random.randint(0, self.max_number_of_slices-1)] for x in range(0, self.max_population)]
        # Актуальная популяция значений хромосом
        self.population_numbers = [self.from_binary_to_number(x) for x in self.population]

    def fitness_function(self, chromosome):
        return self.function(self.from_binary_to_number(chromosome))/sum([self.function(self.from_binary_to_number(x))
                                                                          for x in self.population])

    def selection(self, k=2):
        reselected_population = []
        M = []
        for chromosome in self.population:
            P = self.fitness_function(chromosome)
            for i in range(0, int(abs(P)*self.max_population)):
                M.append(chromosome)
        for i in range(0, self.max_population):
            selected_chromosome = M[random.randint(0, len(M)-1)]
            reselected_population.append(selected_chromosome)
        self.population = reselected_population

    def crossover(self):
        for i in range(0, self.max_population):
            if random.uniform(0, 1) <= self.crossover_chance:
                chrom_a = self.population[random.randint(0, (self.max_population)/2-1)]
                chrom_b = self.population[random.randint((self.max_population)/2, self.max_population-1)]
                self.population.remove(chrom_a)
                self.population.remove(chrom_b)

                k = random.randint(0, self.chromosome_length)

                chrom_a_ = chrom_a[:k] + chrom_b[k:]
                chrom_b_ = chrom_b[:k] + chrom_a[k:]

                self.population.append(chrom_a_)
                self.population.append(chrom_b_)

    def mutation(self):
        for i in range(0, self.max_population):
            if random.uniform(0, 1) <= self.mutation_chance:
                chrom_a = self.population[random.randint(0, self.max_population-1)]
                k = random.randint(0, self.chromosome_length-1)
                temp = list(chrom_a)
                if chrom_a[k] == '0':
                    temp[k] = '1'
                else:
                    temp[k] = '0'
                chrom_a_ = ''.join(temp)
                self.population.remove(chrom_a)
                self.population.append(chrom_a_)

    def plot_graph(self):
        x = np.arange(self.lower_bound, self.upper_bound + 0.1, 0.1)
        y = [self.function(i) for i in x]
        print(max(y))
        x_population = [round(self.from_binary_to_number(x), 3) for x in self.population]
        y_population = [round(self.function(x), 3) for x in x_population]
        plt.plot(x,y)
        plt.scatter(x_population, y_population, alpha=0.5, c=[np.arange(self.max_population)])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'График функции для {self.current_epoch} эпохи')
        plt.grid(True)
        plt.savefig('график.png')

    def write_population_to_file(self):
        self.plot_graph()
        with open('график.png', 'rb') as file:
            image = base64.b64encode(file.read()).decode('utf-8')
        # Добавление графика и описания  на HTML-страницу
        html_content = f'''
            <html>
            <body>
                <h2>График функции {self.current_epoch} эпохи</h2>
                <img src="data:image/png;base64,{image}" alt="График" />
                <p>Настоящее лучшее решение: {self.current_best_solution}</p>
            </body>
            </html>
            '''
        with open('результат.html', 'a') as file:
            file.write(html_content)

    def run(self):
        stop = True
        while stop:
            self.current_epoch += 1
            self.selection()
            self.crossover()
            self.mutation()
            best_solution = max([self.function(self.from_binary_to_number(x)) for x in self.population])
            if round(abs(best_solution - self.current_best_solution), 3) < self.delta:
                stop = False
            elif best_solution > self.current_best_solution:
                self.current_best_solution = best_solution
            if self.current_epoch == self.max_epochs:
                stop = False

            self.write_population_to_file()

def main():
    config = {
        "crossover_chance" : 0.9,
        "mutation_chance" : 0.5,
        "max_population" : 100,
        "max_epochs" : 50,
        "lower_bound" : -20,
        "upper_bound" : -3.1,
    }
    GA = SimpleGA(config)
    GA.run()

if __name__ == '__main__':
    random.seed(1001)
    main()
