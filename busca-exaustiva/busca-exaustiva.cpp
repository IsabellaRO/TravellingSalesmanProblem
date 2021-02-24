#include <iostream>
#include <cmath>
#include <iomanip> 
#include <vector>
#include <algorithm>
#include <random>

using namespace std;

struct city {
    double x;
    double y;
    int id;
};

struct solution {
    double size;
    vector<bool> visited; // já foi visitada ou não
    vector<int> tour; // ordem de visitação
};

double distance(city c1, city c2) {return sqrt(pow(c1.x - c2.x, 2) + pow(c1.y - c2.y, 2));}

void calcula_distancias(vector<double> &mat, vector<city> &cities){
    int n = cities.size();
    for(int i = 0; i < n; i++){
        double dist;
        for(int j = i; j < n; j++){
            dist = distance(cities[i], cities[j]);
	    mat[i*n + j] = dist;
	    mat[j*n + i] = dist;
        }
    }
}

double busca_exaustiva(int i, int N, solution atual, solution &melhor, vector<city> cities, long long int &num_leaf, vector<double> &mat){
    // Caso todas as cidades estejam no tour
    if(atual.tour.size() == N){
        atual.size += mat[i * N + 0];
        num_leaf += 1;
        if(atual.size < melhor.size){
            melhor = atual;
        }
        return melhor.size;
    }

    // Caso a cidade não esteja no tour
    if(atual.visited[i] == false){
        if(atual.tour.empty()){
            atual.size = 0;
        } else{
            atual.size += mat[atual.tour.back() * N + i];
        }
        atual.tour.push_back(i);
        atual.visited[i] = true;

        for(int k = 0; k < N; k++){
            if(atual.visited[k] == false){
                busca_exaustiva(k, N, atual, melhor, cities, num_leaf, mat);
            }
        }
    }
    if(atual.tour.size() == N){
        busca_exaustiva(i, N, atual, melhor, cities, num_leaf, mat);
    }
}

int main() {
    int N, O;
    cin >> N;
    double L;
    L = 0;
    O = 1; 
    
    vector<city> cities(N);

    for(int i = 0; i < N; i++){
        city c;
        cin >> c.x >> c.y;
        c.id = i;
        cities[i] = c;
    }

    solution melhor;
    solution atual;
    melhor.size = 99999.0;
    melhor.visited = vector<bool> (N, false);
    atual.size = 99999.0;
    atual.visited = vector<bool> (N, false);

    long long int num_leaf = 0;
    
    vector<double> mat;
    mat.resize(N*N);
    calcula_distancias(mat, cities);

    double valor = busca_exaustiva(0, N, atual, melhor, cities, num_leaf, mat);
    
    cerr << "num_leaf " << num_leaf << endl;
    cout << valor << " " << O << endl;
    for(auto &c : melhor.tour){
        cout << c << " "; 
    }
    cout << endl;

    return 0;
}
