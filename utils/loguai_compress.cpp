#include<fstream>
#include<iostream>
#include<cassert>
#include"util.h"
using namespace std;

bool compare_matrix(vector<Float>& a, vector<Float>& b){
    if (a.size() != b.size())
        return false;
    for (int i = 0; i < a.size(); i++){
        if (a[i] != b[i]){
            return false;
        }
    }
    return true;
}

vector<vector<Float>> unique_;
int* index_;

int main(int argc, char** argv){
    assert(argc == 3);
    ifstream fin(argv[1]);
    ofstream fout(argv[2]);
    char* line = new char[1000000];
    readLine(fin, line);
    fout << "MARKOV" << endl;
    int T = 0;
    fin >> T;
    fout << T << endl;
    int* T_ = new int[T];
    for (int i = 0; i < T; i++){
        fin >> T_[i];
        fout << T_[i] << " ";
    }
    fout << endl;
    int F = 0;
    fin >> F;
    string* F_ = new string[F];
    index_ = new int[F];
    for (int i = 0; i < F; i++){
        readLine(fin, line);
        F_[i] = string(line);
    }
    for (int i = 0; i < F; i++){
        cerr << unique_.size() << "/" << i << endl;
        int L = -1;
        fin >> L;
        vector<Float> new_mat;
        new_mat.reserve(L);
        for (int j = 0; j < L; j++){
            Float f;
            fin >> f; 
            new_mat.push_back(f);
        }
        index_[i] = unique_.size();
        for (int j = 0; j < unique_.size(); j++){
            if (compare_matrix(unique_[j], new_mat)){
                index_[i] = j;
                break;
            }
        }
        if (index_[i] == unique_.size()){
            unique_.push_back(new_mat);
        }
        cerr << L << "?" << unique_[0].size() << endl;
    }
    cerr << "#uniques=" << unique_.size() << endl;
    fout << unique_.size() << endl; // << " " << F << endl;
    for (int i = 0; i < unique_.size(); i++){
        bool flag = false;
        for (int j = 0; j < F; j++){
            if (index_[j] != i){
                continue;
            }
            if (flag){
                fout << ";";
            }
            fout << F_[j];
            flag = true;
        }
        fout << endl;
    }
    fout << endl;
    for (int i = 0; i < unique_.size(); i++){
        fout << unique_[i].size() << endl;
        for (int j = 0; j < unique_[i].size(); j++){
                fout << unique_[i][j] << " ";
        }
        fout << endl;
        fout << endl;
    }
    //delete F_;
    //delete T_;
    return 0;
}
