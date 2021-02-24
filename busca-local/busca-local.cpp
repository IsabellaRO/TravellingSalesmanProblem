#include <iostream>
#include <cmath>
#include <iomanip> 
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>

using namespace std;

struct city {
    double x;
    double y;
    int id;
    bool in_tour;
};

struct solution {
    int id;
    double size;
    vector<int> result;
};

bool funcValue (solution &s1, solution &s2) { return (s1.size < s2.size); } // checa se s1 é mais curto

double total_dist(vector<city> &order, vector<double> &mat) {
    double dist = 0.0;
    int N = order.size();
    for(int i = 0; i < N; i++){
        if(i == N - 1){
            dist += mat[(order[N-1].id*N)+order[0].id];
        } else{
            dist += mat[(order[i].id*N)+order[i+1].id];
        }
    }
    return dist;
}

double troca_dist(double L, vector<double> &mat, vector<city> &order, int i, int j, int N) {
    int c1 = order[i].id;
    int c2 = order[j].id;

    int jmais;
    (j == N-1 ? jmais = 0 : jmais = j+1);

    int jmenos;
    (j == 0 ? jmenos = N-1 : jmenos = j-1);
    
    int imais;
    (i == N-1 ? imais = 0 : imais = i+1);

    int imenos;
    (i == 0 ? imenos = N-1 : imenos = i-1);

	double new_L = L;
	if((i == 0 && j == N-1) || (i - j == 1)){
		new_L = new_L - mat[(order[jmenos].id*N)+c2] + mat[(order[jmenos].id*N)+c1];
		new_L = new_L - mat[(c1*N)+order[imais].id] + mat[(c2*N)+order[imais].id];
        return new_L;
	} else if((j == 0 && i == N-1) || (j - i == 1)){
		new_L = new_L - mat[(order[imenos].id*N)+c1] + mat[(order[imenos].id*N)+c2];
		new_L = new_L - mat[(c2*N)+order[jmais].id] + mat[(c1*N)+order[jmais].id];
        return new_L;
	} else {
        new_L = new_L - mat[(order[imenos].id*N)+c1] + mat[(order[imenos].id*N)+c2];
        new_L = new_L - mat[(order[jmenos].id*N)+c2] + mat[(order[jmenos].id*N)+c1];
        new_L = new_L - mat[(c1*N)+order[imais].id] + mat[(c2*N)+order[imais].id];
        new_L = new_L - mat[(c2*N)+order[jmais].id] + mat[(c1*N)+order[jmais].id];
        return new_L;
		}
}

void calcula_matriz(vector<double> &mat, vector<city> &cities){
	int N = cities.size();
    for(int i = 0; i < N; i++){
		double dist;
		for(int j = i; j < N; j++){
			double dx = cities[i].x - cities[j].x;
			double dy = cities[i].y - cities[j].y;
			dist = sqrt(pow(dx, 2) + pow(dy, 2));
			mat[i*N + j] = dist;
			mat[j*N + i] = dist;
		}
	}
	return;
}

int main() {
    // PEGA SEED OU SETA COMO DEFAULT
    int seed = 10;
    char *SEED(getenv("SEED"));
    if (SEED != NULL){
        seed = atoi(SEED);
    }
    int debug = 0;
    char *DEBUG(getenv("DEBUG"));
    if (DEBUG != NULL){
        debug = atoi(DEBUG);
    }
    
    // LÊ ENTRADA
    int N;
    cin >> N;
    
    vector<city> cities(N);
    vector<solution> solutions;
    solutions.reserve(10*N);
    vector<double> mat(N*N);

    double L;
    int O = 0;

    for(int i = 0; i < N; i++){
        city c;
        cin >> c.x >> c.y;
        c.id = i;
        c.in_tour = false;
        cities[i] = c;
    }

    calcula_matriz(mat, cities);

    #ifdef _OPENMP
        default_random_engine generator (seed);
        #pragma omp parallel shared(seed, debug, solutions, N) firstprivate(cities, L)
        {
            vector<int> tour;
            tour.reserve(N);
            L = 0;
            
            vector<default_random_engine> generators;
			for(int i = 0; i < omp_get_max_threads(); i++){
				default_random_engine generator (seed + i);
				generators.emplace_back(generator);
			}

            #pragma omp for
            for(int k = 0; k < 10*N; k++){ // FOR DAS SOLUÇÕES
        
                double new_L;
                int O;
                int counter = 0;
                vector<city> order;
                order.reserve(N);
                
                
                while(counter < N){ // Criar tour aleatório
                    for(int i = 0; i < N; i++){
                        uniform_real_distribution<double> distribution(0.0,1.0);
                        double decision = distribution(generators[omp_get_thread_num()]);
                        if(decision >= 0.5 && !cities[i].in_tour){ // entra no tour
                            tour.push_back(cities[i].id);
                            order.push_back(cities[i]);
                            cities[i].in_tour = true;
                            counter++;
                        }
                    }
                }
                
                // REALIZA TROCAS
                for(int i = 0; i < N; i++){
                    for(int j = 0; j < N; j++){
                        if(i != j){
                            city tempi, tempj;
                            vector<city> new_order;
                            new_order = order;
                            new_order[j] = order[i];
                            new_order[i] = order[j];

                            L = total_dist(order, mat);
                            new_L = troca_dist(L, mat, order, i, j, N);
                            if(new_L < L){
                                order[i] = new_order[i];
                                order[j] = new_order[j];
                                tour[i] = new_order[i].id;
                                tour[j] = new_order[j].id;
                                L = new_L;
                                i = 0;
                                j = 0;
                            }
                        }
                    }
                }
                
                solution s; // ADICIONA SOLUÇÃO ATUAL NAS SOLUÇÕES
                s.id = k;
                s.size = L;
                s.result = tour;
                
                #pragma omp critical
                {
                    solutions.push_back(s);
                    if(debug == 1){
                        cerr << "local " << L << " ";
                        for (auto &el : tour){
                            cerr << el << " ";
                        }
                        cerr << "\n";
                    }
                }
                
                // seta cidades como não visitadas novamente
                for(auto &c : cities){
                    c.in_tour = false;
                }
                
                order.clear();
                tour.clear();
            
            }
        }
    #else
        
        default_random_engine generator (seed);
        uniform_real_distribution<double> distribution(0.0,1.0);
        for(int k = 0; k < 10*N; k++){ // FOR DAS SOLUÇÕES
            double new_L;
            int O;
            L = 0;
            int counter = 0;
            vector<city> order;
            order.reserve(N);
            vector<int> tour;
            tour.reserve(N);
            while(counter < N){ // Criar tour aleatório
                for(int i = 0; i < N; i++){
                    double decision = distribution(generator);
                    if(decision >= 0.5 && !cities[i].in_tour){ // entra no tour
                        tour.push_back(cities[i].id);
                        order.push_back(cities[i]);
                        cities[i].in_tour = true;
                        counter++;
                    }
                }
            }
            // REALIZA TROCAS
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    if(i != j){
                        city tempi, tempj;
                        vector<city> new_order;
                        new_order = order;
                        new_order[j] = order[i];
                        new_order[i] = order[j];

                        L = total_dist(order, mat);
                        new_L = troca_dist(L, mat, order, i, j, N);
                        if(new_L < L){
                            order[i] = new_order[i];
                            order[j] = new_order[j];
                            tour[i] = new_order[i].id;
                            tour[j] = new_order[j].id;
                            L = new_L;
                            i = 0;
                            j = 0;
                        }
                    }
                }
            }

            solution s; // ADICIONA SOLUÇÃO ATUAL NAS SOLUÇÕES
            s.id = k;
            s.size = L;
            s.result = tour;
            solutions.push_back(s);
            if(debug == 1){
                cerr << "local " << L << " ";
                for (auto &el : tour){
                    cerr << el << " ";
                }
                cerr << "\n";
            }

            // seta cidades como não visitadas novamente
            for(auto &c : cities){
                c.in_tour = false;
            }

            order.clear();
            tour.clear();
        
        }
    #endif
    // Encotra a melhor das soluções
    
    sort(solutions.begin(), solutions.end(), funcValue); // ordena de acordo com mais curto
    double min_dist = solutions[0].size;
    vector<int> result = solutions[0].result;
    cout << min_dist << " " << O << endl;
    for(auto c : result){
        cout << c << " "; 
    }
    cout << endl;

    return 0;
}
