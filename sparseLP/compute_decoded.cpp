#include<fstream>
#include"problem.h"
#include<string>
#include<iostream>

using namespace std;

char* line = new char[LINE_LEN];

int main(int argc, char** argv){
    ifstream uai(argv[1]);
    ifstream sol(argv[2]);

    int T;
    sol >> T;
    readLine(sol, line);
    sol.close();

    string line_str(line);
    vector<string> tokens = split(line_str, " ");

    readLine(uai, line);
    uai >> T;
    cerr << "T=" << T << endl; // << ", tokens.size=" << tokens.size() << endl;
    assert(T == tokens.size());
    int* pred = new int[T];
    for (int i = 0; i < T; i++){
        pred[i] = stoi(tokens[i]);
    }
    int* K_ = new int[T];
    for (int i = 0; i < T; i++){
        uai >> K_[i];
    }
    int F;
    uai >> F;
    cerr << "F=" << F << endl;
    F = F - T;
    for (int i = 0; i < T; i++){
        int d = 0, l;
        uai >> d;
        assert(d == 1);
        uai >> l;
        assert(l == i);
    }
    int* l = new int[F];
    int* r = new int[F];
    for (int i = 0; i < F; i++){
        int d = 0;
        uai >> d;
        assert(d == 2);
        uai >> l[i] >> r[i];
        assert(l[i] < r[i]);
    }
    
    Float final = 0.0;
    for (int i = 0; i < T; i++){
        int s;
        uai >> s;
        assert(s == K_[i]);
        for (int j = 0; j < K_[i]; j++){
            Float f;
            uai >> f;
            if (j == pred[i]){
                final += -log(f);
            }
        }
    }
    for (int i = 0; i < F; i++){
        int s;
        uai >> s;
        assert(s == K_[l[i]] * K_[r[i]] );
        for (int j = 0; j < s; j++){
            Float f;
            uai >> f;
            if (j == (pred[l[i]]*K_[r[i]] + pred[r[i]]) ){
                final += -log(f);
            }
        }
    }

    cerr << "score=" << final << endl;

    uai.close();

}
