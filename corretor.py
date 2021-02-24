#coding: utf-8

from grading_tools import TestConfiguration, ProgramTest, CheckOutputMixin, CheckStderrMixin, CheckMultiCorePerformance
import sys
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import os
import io
import os.path as osp
import sys
from contextlib import redirect_stdout
import numba as nb
from numba.typed import List as nbList
import pprint

from colorama import init, Fore

def green(s):
    return Fore.GREEN + str(s) + Fore.RESET

def red(s):
    return Fore.RED + str(s) + Fore.RESET

def test_result(b):
    if b:
        return green(b)
    else:
        return red(b)

class BaseTSP:
    def parse_input(self, inp):
        lines = inp.split('\n')
        N = int(lines[0])
        points = np.zeros((N, 2))
        for i, l in enumerate(lines[1:-1]):
            p = l.split()
            points[i, 0] = float(p[0])
            points[i, 1] = float(p[1])

        distances = euclidean_distances(points)
        return N, points, distances

    def parse_output(self, stdout):
        try:
            lines = stdout.split('\n')
            distance, opt = lines[0].split()
            distance = float(distance)
            order = [int(i) for i in lines[1].split()]
        except:
            for i, l in enumerate(lines):
                print('linha', i, ':', l)
            raise Exception("Formato de saída inválido")

        return distance, opt, order

    def tamanho_tour(self, order, distances):
        tour = 0
        last = order[0]
        N = len(order)
        for i in range(1, N):
            tour += distances[last, order[i]]
            last = order[i]
        tour += distances[order[N-1], order[0]]

        return tour

    def test_caminho_tem_distancia_mostrada(self, test, stdout, stderr):
        N, points, distances = self.parse_input(test.input)
        distance, opt, order = self.parse_output(stdout)
        tour = self.tamanho_tour(order, distances)
        res = np.isclose(tour, distance)
        if not res:
            print('Distância calculada:', tour, 'Mostrada:', distance)
        return res

    def test_passa_por_todas_cidades(self, test, stdout, stderr):
        lines = stdout.split('\n')
        cities = [int(i) for i in lines[1].split()]
        cities.sort()
        return cities == list(range(len(cities)))


@nb.jit
def tamanho_tour_nb(order, distances):
        tour = 0
        last = order[0]
        N = len(order)
        for i in range(1, N):
            tour += distances[last, order[i]]
            last = order[i]
        tour += distances[order[N-1], order[0]]

        return tour

def tem_troca_py(order, distance, distances):
    N = len(order)
    for i in range(N):
        for j in range(i+1, N):
            new_order = order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_distance = tamanho_tour_nb(new_order, distances)
            is_close = abs(distance - new_distance)
            if is_close > 0.01 and distance > new_distance:
                return True, i, j, new_distance

    return False, 0, 0, 0.0

tem_troca_nb = nb.jit(tem_troca_py)

class TesteBuscaLocal(ProgramTest, BaseTSP):
    def tem_troca(self, order, distance, distances):
        distance = self.tamanho_tour(order, distances)
        ret, i, j, new_distance = tem_troca_nb(nbList(order), distance, distances)
        if ret:
            print('Troca encontrada: ', i, j)
            print('Melhoria:', distance, '->', new_distance)
        return ret

    def test_nao_tem_troca(self, test, stdout, stderr):
        N, points, distances = self.parse_input(test.input)
        distance, opt, order = self.parse_output(stdout)

        return not self.tem_troca(order, distance, distances)

    def parse_stderr(self, stderr):
        lines = stderr.split('\n')
        solutions = []
        for l in lines:
            l = l.strip()
            try:
                distance = float(l.split(' ')[1])
            except:
                print(l.split())
                raise Exception('Formato de saída de erros inválido')
            try:
                order = [int(i) for i in l.split(' ')[2:]]
            except:
                print(l.split(' ')[2:])
                raise Exception('Formato de saída de erros inválido')

            solutions.append((distance, order))

        return solutions

    def test_fez_10N_solucoes(self, test, stdout, stderr):
        N, points, distances = self.parse_input(test.input)
        distance, opt, order = self.parse_output(stdout)
        solutions = self.parse_stderr(stderr)
        return len(solutions) == 10 * N

    def test_toda_solucao_tem_caminho_correto(self, test, stdout, stderr):
        N, points, distances = self.parse_input(test.input)
        distance, opt, order = self.parse_output(stdout)
        solutions = self.parse_stderr(stderr)
        for i, sol in enumerate(solutions):
            dist_calculada = self.tamanho_tour(sol[1], distances)
            if not np.isclose(dist_calculada, sol[0]):
                print(f'Caminho incorreto na linha {i}. Mostrada {sol[0]}, calculada {dist_calculada}')
                print(sol[1], self.tamanho_tour(sol[1], distances), tamanho_tour_nb(sol[1], distances))
                return False
        return True

    def test_toda_solucao_eh_otimo_local(self, test, stdout, stderr):
        N, points, distances = self.parse_input(test.input)
        distance, opt, order = self.parse_output(stdout)
        solutions = self.parse_stderr(stderr)
        for i, sol in enumerate(solutions):
            if self.tem_troca(sol[1], sol[0], distances):
                print('Troca acima encontrada na linha', i)
                return False
        return True


class TesteHeuristico(ProgramTest, CheckOutputMixin, BaseTSP):
    pass


class TesteBuscaExaustiva(ProgramTest, CheckStderrMixin, BaseTSP):
    def test_tour_otimo(self, test, stdout, stderr):
        N, points, distances = self.parse_input(test.input)
        distance, opt, order = self.parse_output(stdout)
        correct_distance, correct_opt, correct_order = self.parse_output(test.output)
        return np.isclose(correct_distance, distance)

    def test_opt_1(self, test, stdout, stderr):
        distance, opt, order = self.parse_output(stdout)
        return int(opt) == 1

class TesteBuscaExaustivaPerf(ProgramTest, BaseTSP):
    def test_tour_otimo(self, test, stdout, stderr):
        N, points, distances = self.parse_input(test.input)
        distance, opt, order = self.parse_output(stdout)
        correct_distance, correct_opt, correct_order = self.parse_output(test.output)
        return np.isclose(correct_distance, distance)

    def test_opt_1(self, test, stdout, stderr):
        distance, opt, order = self.parse_output(stdout)
        return int(opt) == 1


class TestePerformance(ProgramTest, BaseTSP):
    pass

class TesteMultiCorePequeno(TesteBuscaLocal, CheckMultiCorePerformance):
    pass

def compila_programa(ext, nome, flags, nvcc=False):
    compilador = 'g++'
    if nvcc:
        compilador = 'nvcc'
    arquivos = ' '.join([arq for arq in os.listdir('.') if arq.split('.')[-1] == ext])
    ret = os.system(f'{compilador} {arquivos} -O3 {flags} -o {nome} 2>1 > /dev/null')
    if ret != 0 :
        raise IOError(f'Erro de compilação em {os.getcwd()}!')

def testa_heuristico():
    os.chdir('heuristico')
    compila_programa('cpp', 'heuristico', '')
    tests = TestConfiguration.from_pattern('.', 'in*.txt', 'out*txt')
    tester = TesteHeuristico('./heuristico', tests)
    res = tester.main()
    os.chdir('..')
    return res


def testa_busca_local_sequencial():
    os.chdir('busca-local')
    compila_programa('cpp', 'busca-local', '')
    tests = TestConfiguration.from_pattern('.', 'in*.txt', 'out*txt',
                                           environ={
                                               'DEBUG': '1'
                                           }, time_limit=1800)
    tester = TesteBuscaLocal('./busca-local', tests)
    res = tester.main()
    os.chdir('..')
    return res

def testa_busca_exaustiva():
    os.chdir('busca-exaustiva')
    compila_programa('cpp', 'busca-global', '')
    tests = TestConfiguration.from_pattern('.', 'in*.txt', 'out*txt',
                                           'err*txt', environ={
                                               'DEBUG': '1'
                                           })
    tester = TesteBuscaExaustiva('./busca-global', tests)
    res = tester.main()
    os.chdir('..')
    return res

def testa_busca_local_omp():
    os.chdir('busca-local')
    compila_programa('cpp', 'busca-local-paralela', '-fopenmp')
    tests = TestConfiguration.from_pattern('.', 'in*.txt', 'out*txt',
                                           environ={
                                               'DEBUG': '1'
                                           }, time_limit=1800)
    files_multicore = [f'./in-{i}.txt' for i in range(5, 10)]
    omp_tests = {k:v for k, v in tests.items() if k in files_multicore}
    del omp_tests['./in-7.txt']
    del omp_tests['./in-8.txt']
    
    teste_simples = TesteMultiCorePequeno('./busca-local-paralela', omp_tests)
    res = teste_simples.main()
    os.chdir('..')
    return res

def testa_busca_local_gpu():
    os.chdir('busca-local')
    compila_programa('cu', 'busca-local-gpu', '', True)
    tests = TestConfiguration.from_pattern('.', 'in*.txt', 'out*txt',
                                        environ={
                                            'DEBUG': '1'
                                        })
    tester = TesteBuscaLocal('./busca-local-gpu', tests)

    perf_tests = {}
    perf_tests['in6.txt'] = TestConfiguration.from_file('in-6.txt',
                                                '',
                                                check_stderr=False,
                                                environ={'DEBUG': '0'},
                                                time_limit=30)
    perf_tests['in11.txt'] = TestConfiguration.from_file('perf-in-11.txt',
                                                '',
                                                check_stderr=False,
                                                environ={'DEBUG': '0'},
                                                time_limit=30)

    perf_tester = TestePerformance('./busca-local-gpu', perf_tests)
    res = tester.main() and perf_tester.main()
    os.chdir('..')
    return res

def testa_busca_local_gpu2():
    os.chdir('busca-local')
    compila_programa('cu', 'busca-local-gpu2', '', True)
    tests = TestConfiguration.from_pattern('.', 'in*.txt', 'out*txt',
                                        environ={
                                            'DEBUG': '1'
                                        })
    tester = TesteBuscaLocal('./busca-local-gpu2', tests)

    perf_tests = {}
    perf_tests['in6.txt'] = TestConfiguration.from_file('in-6.txt',
                                                '',
                                                check_stderr=False,
                                                environ={'DEBUG': '0'},
                                                time_limit=10)
    perf_tests['in11.txt'] = TestConfiguration.from_file('perf-in-11.txt',
                                                    '',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=9)

    perf_tests['in12.txt'] = TestConfiguration.from_file('perf-in-12.txt',
                                                    '',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=65)

    perf_tester = TestePerformance('./busca-local-gpu2', perf_tests)
    res = tester.main() and perf_tester.main()
    os.chdir('..')
    return res

def testa_busca_local_perf():
    os.chdir('busca-local')
    compila_programa('cpp', 'busca-local-perf', '')
    performance_tests = {}
    performance_tests['in10.txt'] = TestConfiguration.from_file('in-10.txt',
                                                    '',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=0.5)

    performance_tests['in11.txt'] = TestConfiguration.from_file('perf-in-11.txt',
                                                    '',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=9)

    performance_tests['in12.txt'] = TestConfiguration.from_file('perf-in-12.txt',
                                                    '',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=180)

    perf_tester = TestePerformance('./busca-local-perf', performance_tests)
    res = perf_tester.main()
    os.chdir('..')
    return res 

def testa_busca_exaustiva_perf1():
    os.chdir('busca-exaustiva')
    compila_programa('cpp', 'busca-global-perf', '-fopenmp')
    performance_tests = {}
    performance_tests['in-5.txt'] = TestConfiguration.from_file('in-5.txt',
                                                    'out-5.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=1)
    performance_tests['in-6.txt'] = TestConfiguration.from_file('in-6.txt',
                                                    'out-6.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=1)
    performance_tests['in-7.txt'] = TestConfiguration.from_file('in-7.txt',
                                                    'out-7.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=5)

    perf_tester = TesteBuscaExaustivaPerf('./busca-global-perf', performance_tests)
    res = perf_tester.main()
    os.chdir('..')
    return res

def testa_busca_exaustiva_perf2():
    os.chdir('busca-exaustiva')
    compila_programa('cpp', 'busca-global-perf2', '-fopenmp')
    performance_tests = {}
    performance_tests['in-7.txt'] = TestConfiguration.from_file('in-7.txt',
                                                    'out-7.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=1)
    performance_tests['perf-in-8.txt'] = TestConfiguration.from_file('perf-in-8.txt',
                                                    'perf-out-8.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=2)
    performance_tests['perf-in-9.txt'] = TestConfiguration.from_file('perf-in-9.txt',
                                                    'perf-out-9.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=30)

    perf_tester = TesteBuscaExaustivaPerf('./busca-global-perf2', performance_tests)
    res = perf_tester.main()
    os.chdir('..')
    return res

def testa_busca_exaustiva_perf3():
    os.chdir('busca-exaustiva')
    compila_programa('cpp', 'busca-global-perf3', '-fopenmp')
    performance_tests = {}
    performance_tests['in-7.txt'] = TestConfiguration.from_file('in-7.txt',
                                                    'out-7.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=1)
    performance_tests['perf-in-8.txt'] = TestConfiguration.from_file('perf-in-8.txt',
                                                    'perf-out-8.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=1)
    performance_tests['perf-in-9.txt'] = TestConfiguration.from_file('perf-in-9.txt',
                                                    'perf-out-9.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=10)
    performance_tests['perf-in-10.txt'] = TestConfiguration.from_file('perf-in-10.txt',
                                                    'perf-out-10.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=25)

    perf_tester = TesteBuscaExaustivaPerf('./busca-global-perf3', performance_tests)
    res = perf_tester.main()
    os.chdir('..')
    return res

def testa_busca_exaustiva_perf4():
    os.chdir('busca-exaustiva')
    compila_programa('cpp', 'busca-global-perf4', '-fopenmp')
    performance_tests = {}
    performance_tests['in-7.txt'] = TestConfiguration.from_file('in-7.txt',
                                                    'out-7.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=1)
    performance_tests['perf-in-8.txt'] = TestConfiguration.from_file('perf-in-8.txt',
                                                    'perf-out-8.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=1)
    performance_tests['perf-in-9.txt'] = TestConfiguration.from_file('perf-in-9.txt',
                                                    'perf-out-9.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=2)
    performance_tests['perf-in-10.txt'] = TestConfiguration.from_file('perf-in-10.txt',
                                                    'perf-out-10.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=3)
    performance_tests['perf-in-11.txt'] = TestConfiguration.from_file('perf-in-11.txt',
                                                    'perf-out-11.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=3)
    perf_tester = TesteBuscaExaustivaPerf('./busca-global-perf4', performance_tests)
    res = perf_tester.main()
    os.chdir('..')
    return res

def testa_busca_exaustiva_perf5():
    os.chdir('busca-exaustiva')
    compila_programa('cpp', 'busca-global-perf5', '-fopenmp')
    performance_tests = {}
    performance_tests['in-7.txt'] = TestConfiguration.from_file('in-7.txt',
                                                    'out-7.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=1)
    performance_tests['perf-in-8.txt'] = TestConfiguration.from_file('perf-in-8.txt',
                                                    'perf-out-8.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=1)
    performance_tests['perf-in-9.txt'] = TestConfiguration.from_file('perf-in-9.txt',
                                                    'perf-out-9.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=2)
    performance_tests['perf-in-10.txt'] = TestConfiguration.from_file('perf-in-10.txt',
                                                    'perf-out-10.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=3)
    performance_tests['perf-in-11.txt'] = TestConfiguration.from_file('perf-in-11.txt',
                                                    'perf-out-11.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=3)
    performance_tests['perf-in-12.txt'] = TestConfiguration.from_file('perf-in-12.txt',
                                                    'perf-out-12.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=45)
    performance_tests['perf-in-13.txt'] = TestConfiguration.from_file('perf-in-13.txt',
                                                    'perf-out-13.txt',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=70)

    perf_tester = TesteBuscaExaustivaPerf('./busca-global-perf5', performance_tests)
    res = perf_tester.main()
    os.chdir('..')
    return res


def testa_busca_local_omp_perf():
    os.chdir('busca-local')
    compila_programa('cpp', 'busca-local-perf-omp', '-fopenmp')
    performance_tests = {}
    performance_tests['in10.txt'] = TestConfiguration.from_file('in-10.txt',
                                                    '',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=0.5)

    performance_tests['in11.txt'] = TestConfiguration.from_file('perf-in-11.txt',
                                                    '',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=3)

    performance_tests['in12.txt'] = TestConfiguration.from_file('perf-in-12.txt',
                                                    '',
                                                    check_stderr=False,
                                                    environ={'DEBUG': '0'},
                                                    time_limit=60)

    perf_tester = TestePerformance('./busca-local-perf-omp', performance_tests)
    res = perf_tester.main()
    os.chdir('..')
    return res


if __name__ == "__main__":

    ignorar = io.StringIO()

    testesD = {
        'heuristico': ('Heuristico (sequencial)', testa_heuristico),
        'local': ('Busca local (sequencial)', testa_busca_local_sequencial),
        'global': ('Busca exaustiva (sequencial)', testa_busca_exaustiva),
        'local-paralela': ('Busca local (paralela)', testa_busca_local_omp),
        'local-gpu': ('Busca local (GPU)', testa_busca_local_gpu)
    }

    if len(sys.argv) > 1:
        tst = sys.argv[1]
        if tst in testesD:
            tst = testesD[tst]
            print(tst[0], ':', tst[1]())
        else:
            print('Testes disponíveis:')
            pprint.pprint(testesD)
        sys.exit(0)

    print(f'''
===========================================
Rubrica D
===========================================''')

    with open('feedback-heuristico.txt', 'w') as f:
        with redirect_stdout(f):
            res_heuristico = testa_heuristico()

    print(f'Heuristico (sequencial): {test_result(res_heuristico)}')

    
    with open('feedback-local.txt', 'w') as f:
        with redirect_stdout(f):
            res_busca_local_sequencial = testa_busca_local_sequencial()

    print(f'Busca local (sequencial): {test_result(res_busca_local_sequencial)}')

    with open('feedback-global.txt', 'w') as f:
        with redirect_stdout(f):
            res_global = testa_busca_exaustiva()
    print(f'Busca exaustiva (sequencial): {test_result(res_global)}')

    with open('feedback-local-paralela.txt', 'w') as f:
        with redirect_stdout(f):
            res_busca_local_paralela = testa_busca_local_omp()

    print(f'Busca local (paralela): {test_result(res_busca_local_paralela)}')
    
    with open('feedback-local-gpu.txt', 'w') as f:
        with redirect_stdout(f):
            res_busca_local_gpu = testa_busca_local_gpu()

    print(f'Busca local (GPU): {test_result(res_busca_local_gpu)}')
    
    print('''===========================================
Rubrica A+
===========================================''')

    with open('feedback-local-perf.txt', 'w') as f:
        with redirect_stdout(f):
            res_busca_local_perf = testa_busca_local_perf()

    print(f'Busca local (desempenho sequencial): {test_result(res_busca_local_perf)}')

    with open('feedback-local-perf-omp.txt', 'w') as f:
        with redirect_stdout(f):
            res_busca_local_omp_perf = testa_busca_local_omp_perf()

    print(f'Busca local (desempenho paralelo 1): {test_result(res_busca_local_omp_perf)}')

    with open('feedback-local-perf-gpu2.txt', 'w') as f:
        with redirect_stdout(f):
            res_busca_local_perf_gpu = testa_busca_local_gpu2()

    print(f'Busca local (desempenho GPU 1): {test_result(res_busca_local_perf_gpu)}')


    with open('feedback-global-perf.txt', 'w') as f:
        with redirect_stdout(f):
            res_busca_global_p1 = testa_busca_exaustiva_perf1()

    print(f'Busca global (desempenho nível 1): {test_result(res_busca_global_p1)}')

    with open('feedback-global-perf2.txt', 'w') as f:
        with redirect_stdout(f):
            res_busca_global_p2 = testa_busca_exaustiva_perf2()

    print(f'Busca global (desempenho nível 2): {test_result(res_busca_global_p2)}')

    with open('feedback-global-perf3.txt', 'w') as f:
        with redirect_stdout(f):
            res_busca_global_p3 = testa_busca_exaustiva_perf3()

    print(f'Busca global (desempenho nível 3): {test_result(res_busca_global_p3)}')

    with open('feedback-global-perf4.txt', 'w') as f:
        with redirect_stdout(f):
            res_busca_global_p4 = testa_busca_exaustiva_perf4()

    print(f'Busca global (desempenho nível 4): {test_result(res_busca_global_p4)}')

    with open('feedback-global-perf5.txt', 'w') as f:
        with redirect_stdout(f):
            res_busca_global_p5 = testa_busca_exaustiva_perf5()

    print(f'Busca global (desempenho nível 5): {test_result(res_busca_global_p5)}')
