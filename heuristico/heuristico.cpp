#include <iostream>
#include <cmath>
#include <iomanip> 
#include <vector>
#include <algorithm>

using namespace std;

struct city {
    double x;
    double y;
    int id;
    bool visited;
    double distance;
};

double dist(city c1, city c2) {return sqrt(pow(c1.x - c2.x, 2) + pow(c1.y - c2.y, 2));}
double closest(city c1, city c2) {return c1.distance < c2.distance;}

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
    int N, O;
    cin >> N;
    double L;
    L = 0;
    O = 0; 
    
    vector<city> cities(N);
    vector<double> mat(N*N);

    for(int i = 0; i < N; i++){
        city c;
        cin >> c.x >> c.y;
        c.id = i;
        c.visited = false;
        c.distance;
        cities[i] = c;
    }

    calcula_matriz(mat, cities);

    city start = cities[0];
    city current = cities[0];
    int counter;
    vector<int> tour;
    tour.push_back(start.id);
    cities[0].visited = true;
    for(int i = 1; i < N; i++){ // i é cidades visitadas - primeira cidade
        for(int j = 0; j < N; j++){
            if(cities[j].visited == false){ // não é necessário mas pode diminuir tempo...
                cities[j].distance = mat[(current.id*N)+cities[j].id];
            }
        }

        sort(cities.begin(), cities.end(), closest);
        counter = 0;
        for(auto c : cities){
            if(c.visited == false){
                current = c;
                break;
            }
            counter++;
        }
        tour.push_back(current.id);
        L = L + current.distance;
        cities[counter].visited = true;
    }

    L = L + mat[(cities[counter].id*N)+start.id];

    cout << L << " " << O << endl;
    for(auto c : tour){
        cout << c << " "; 
    }
    cout << endl;

    return 0;
}