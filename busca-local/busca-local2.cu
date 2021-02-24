#include <iostream>
#include <cmath>
#include <iomanip> 
#include <vector>
#include <algorithm>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/random.h>

using namespace std;

__host__ __device__
double total_dist(int *order, double *mat, int N, int k) {
    double dist = 0.0;
    for(int i = 0; i < N; i++){
        if(i == N - 1){
            dist += mat[(order[(N*k)+N-1]*N)+order[(N*k)+0]];
        } else{
            dist += mat[(order[(N*k)+i]*N)+order[(N*k)+i+1]];
        }
    }
    return dist;
}

__host__ __device__
double troca_dist(double L, double *mat, int *order, int i, int j, int N, int k) {
    int c1 = order[(N*k)+i];
    int c2 = order[(N*k)+j];

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
		new_L = new_L - mat[(order[(N*k)+jmenos]*N)+c2] + mat[(order[(N*k)+jmenos]*N)+c1];
		new_L = new_L - mat[(c1*N)+order[(N*k)+imais]] + mat[(c2*N)+order[(N*k)+imais]];
        return new_L;
	} else if((j == 0 && i == N-1) || (j - i == 1)){
		new_L = new_L - mat[(order[(N*k)+imenos]*N)+c1] + mat[(order[(N*k)+imenos]*N)+c2];
		new_L = new_L - mat[(c2*N)+order[(N*k)+jmais]] + mat[(c1*N)+order[(N*k)+jmais]];
        return new_L;
	} else {
        new_L = new_L - mat[(order[(N*k)+imenos]*N)+c1] + mat[(order[(N*k)+imenos]*N)+c2];
        new_L = new_L - mat[(order[(N*k)+jmenos]*N)+c2] + mat[(order[(N*k)+jmenos]*N)+c1];
        new_L = new_L - mat[(c1*N)+order[(N*k)+imais]] + mat[(c2*N)+order[(N*k)+imais]];
        new_L = new_L - mat[(c2*N)+order[(N*k)+jmais]] + mat[(c1*N)+order[(N*k)+jmais]];
        return new_L;
	}
}

struct raw_access {
    int *order;
    int *new_order;
    int *solutions;
    int seed;
    int debug;
    int N;
    double *L;
    bool *in_tour;
    double *mat;

    raw_access (int *order, int *new_order, int *solutions, int seed, int debug, int N, double *L, bool *in_tour, double *mat) : order(order), new_order(new_order), solutions(solutions), seed(seed), debug(debug), N(N), L(L), in_tour(in_tour), mat(mat) {};
    
    __host__ __device__
    double operator()(const int &i) {
        int counter = 0;
        
        thrust::default_random_engine generator(seed);

        while(counter < N){ // Criar tour aleatório
            for(int k = 0; k < N; k++){
                
                generator.discard(i);
                thrust::uniform_real_distribution<float> distribution(0, 1);
                double decision = distribution(generator);
                
                if(decision >= 0.5 && !in_tour[(i*N)+k]){ // entra no tour
                    order[(i*N)+counter] = k;
                    in_tour[(i*N)+k] = true;
                    counter++;
                }
            }
        }

        for(int h = 0; h < N; h++){
            for(int j = 0; j < N; j++){
                if(h != j){
                    for(int l = 0; l < N; l++){
                        new_order[(i*N)+l] = order[(i*N)+l];
                    }
                    
                    new_order[(i*N)+j] = order[(i*N)+h];
                    new_order[(i*N)+h] = order[(i*N)+j];
                    
                    L[i] = total_dist(order, mat, N, i);
                    double new_L = troca_dist(L[i], mat, order, h, j, N, i);
                    
                    if(new_L < L[i]){
                        order[(i*N)+h] = new_order[(i*N)+h];
                        order[(i*N)+j] = new_order[(i*N)+j];
                        L[i] = new_L;
                        h = 0;
                        j = 0;
                    }
                }
            }
        }

        // ADICIONA SOLUÇÃO ATUAL NAS SOLUÇÕES
        for(int k = 0; k < N; k++){
            solutions[(i*N)+k] = order[(i*N)+k];
        }

        return L[i];
    }
};

int main() {
    // PEGA SEED E DEBUG OU SETA COMO DEFAULT
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
    
    thrust::device_vector<int> solutions(10*N*N); // tours possíveis
    thrust::device_vector<double> distancias(10*N); // distancias de cada tour
    thrust::device_vector<double> L(10*N, 0.0); // um pra cada thread
    int O = 0;

    thrust::device_vector<double> X(N);
    thrust::device_vector<double> Y(N);
    
    for(int i = 0; i < N; i++){
        double x, y;
        cin >> x >> y;
        X[i] = x;
        Y[i] = y;
    }

    thrust::device_vector<double> mat(N*N);
    for(int i = 0; i < N; i++){
        double dist;
        for(int j = i; j < N; j++){
            double dx = X[i] - X[j];
            double dy = Y[i] - Y[j];
            dist = sqrt(pow(dx, 2) + pow(dy, 2));
            mat[i*N + j] = dist;
            mat[j*N + i] = dist;
        }
    }

    thrust::device_vector<bool> in_tour (10*N*N, false);
    thrust::device_vector<int> order(10*N*N);
    thrust::device_vector<int> new_order(10*N*N);

    thrust::counting_iterator<int> iter(0);
    raw_access ra(thrust::raw_pointer_cast(order.data()), thrust::raw_pointer_cast(new_order.data()), thrust::raw_pointer_cast(solutions.data()), seed, debug, N, thrust::raw_pointer_cast(L.data()), thrust::raw_pointer_cast(in_tour.data()), thrust::raw_pointer_cast(mat.data()) );
    thrust::transform(iter, iter+distancias.size(), distancias.begin(), ra);

    double menordist = 100000;
    int position = -1;
    for(int i = 0; i < 10*N; i++){
        if(distancias[i] < menordist){
            menordist = distancias[i];
            position = i;
        }
    }


    //thrust::copy(solutions.begin(), solutions.end(), solucoes);
    thrust::host_vector<int> solucoes(10*N);
    solucoes = solutions;

    thrust::host_vector<int> ordens(10*N*N);
    ordens = order;

    thrust::host_vector<int> solucao_final(N);
    for(int i = 0; i < N; i++){
        solucao_final[i] = solucoes[(position*N)+i];
    }

    if(debug == 1){
        for(int k = 0; k < 10*N; k++){
            cerr << "local " << distancias[k];
            for (int m = 0; m < N; m++){
                cerr << " " << ordens[(k*N)+m];
            }
            cerr << "\n";
        }
    }

    cout << menordist << " " << O << endl;
    for(int i = 0; i < N; i++){
        cout << solucao_final[i] << " ";
    }
    cout << endl;

    return 0;
}

