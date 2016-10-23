#include<fstream>
#include<iostream>
#include<cassert>
#include"util.h"
#include"loguai_compress.h"

using namespace std;

vector<Factor> factor;
//vector<Function> function;
vector<Var> var;

int main(int argc, char** argv){
	assert(argc == 3);
	ifstream fin(argv[1]);
	char* line = new char[1000000];
	readLine(fin, line);
	int T = 0;
	fin >> T;
	for (int i = 0; i < T; i++){
		int size;
		fin >> size;
		var.push_back(Var(i, size)); // id, size
	}
	int num_fac;
	fin >> num_fac;
	cerr << "num_fac=" << num_fac << endl;
	int* corresponding_factor = new int[num_fac];
	for (int fac = 0; fac < num_fac; fac++){
		int D;
		fin >> D;
		assert(D == 1 || D == 2);
		Factor f;
		if (D == 1){
			int i;
			fin >> i;
			f = Factor(var[i]);
		} else {
			int i, j;
			fin >> i >> j;
			f = Factor(var[i], var[j]);
		}
		corresponding_factor[fac] = factor.size();
		for (int i = 0; i < factor.size(); i++){
			if (factor[i].equalto(f)){
				corresponding_factor[fac] = i;
			}
		}
		if (corresponding_factor[fac] == factor.size()){
			//cerr << "pushing in" << endl;
			factor.push_back(f);
		}
		//cerr << "corr=" << corresponding_factor[fac] << endl;
		//f.print();
	}
	for (int fac = 0; fac < num_fac; fac++){
		int L = -1;
		fin >> L;
		//cerr << "get\t" << L << endl;
		//cerr << "updating factor #" << corresponding_factor[fac] << endl;
		Factor& f = factor[corresponding_factor[fac]];
		for (int j = 0; j < L; j++){
			Float val;
			fin >> val;
			//cerr << "got\t" << val << endl;
			f.function->score_vec[j] += val;
			//cerr << "factor.size=" << factor.size() << endl;
			//cerr << "==================f[0]====================" << endl;
			//factor[0].print();
			//cerr << "==================f[0]====================" << endl;
			//cerr << "==================f[1]====================" << endl;
			//factor[1].print();
			//cerr << "==================f[1]====================" << endl;
			//cerr << "==================f[2]====================" << endl;
			//factor[2].print();
			//cerr << "==================f[2]====================" << endl;
		}
		//f.print();
		//cerr << "==================f[0]====================" << endl;
		//factor[0].print();
		//cerr << "==================f[0]====================" << endl;
	}
	cerr << "closing input file" << endl;
	fin.close();
	vector<vector<Factor*>> factor_block;
	for (int fac = 0; fac < factor.size(); fac++){
		Factor* f = &(factor[fac]);
		bool flag = false;
		for (int j = 0; j < factor_block.size(); j++){
			//cerr << "getting representative....";
			Factor* block_rep = factor_block[j][0];
			//cerr << "done" << endl;
			if (f->has_same_score_vec(block_rep)){
				//add to existing block
				//cerr << "appending function...";
				factor_block[j].push_back(f);
				flag = true;
				//cerr << "done" << endl;
				break;
			}
		}
		if (!flag){
			//add function to new block
			//cerr << "adding function...";
			vector<Factor*> new_block;
			factor_block.push_back(new_block);
			factor_block[factor_block.size()-1].push_back(f);
			//cerr << "done" << endl;
		}
	}
	
	cerr << "#uniques=" << factor_block.size() << "(out of " << num_fac << ")" << endl;
	//fout << num_fac << endl;
	ofstream fout(argv[2]);
	fout << "MARKOV" << endl;
	fout << var.size() << endl;
	for (int i = 0; i < var.size(); i++){
		fout << var[i].size << " ";
	}
	fout << endl;
	fout << factor.size() << endl;
	for (int b = 0; b < factor_block.size(); b++){
		for (int j = 0; j < factor_block[b].size(); j++){
			Factor* f = factor_block[b][j];
			if (j == 0){
				fout << f->vars.size();
			} else {
				fout << -((int)(f->vars.size()));
			}
			for (int i = 0; i < f->vars.size(); i++){
				fout << " " << f->vars[i].id;
			}
			fout << endl;
		}
	}
	cerr << "done writing preambles" << endl;
	fout << endl;
	for (int b = 0; b < factor_block.size(); b++){
		Factor* fb = factor_block[b][0];
		//output its score_vec
		fout << fb->size() << endl;
		for (int l = 0;  l < fb->size(); l++){
			fout << fb->function->score_vec[l] << " ";
		}
		fout << endl;
		fout << endl;
	}
	fout.close();
	
	return 0;
}
