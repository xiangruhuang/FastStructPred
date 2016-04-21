#ifndef PROBLEM_H
#define PROBLEM_H 

#include "util.h"
#include "extra.h"
#include <cassert>

//each Instance is an isolated subgraph
class Instance{
	public:
	vector<SparseVec*> features; 
	Labels labels; 
	Int T; // number of nodes
	vector<pair<Int, Int>> edges; // edge list of this subgraph
	vector<vector<Int>*> adj; // adjacency list of each node, only store edges towards nodes of larger id
	void build_adj(){
		//assume edges is constructed, build adjacency list
		adj.clear();
		for (int i = 0; i < T; i++){
			adj.push_back(new vector<Int>());
		}
		for (vector<pair<Int, Int>>::iterator e = edges.begin(); e != edges.end(); e++){
			assert(e->first < e->second /* need i < j for each edge (i,j) */);
			adj[e->first]->push_back(e->second);
		}
	}
};

class Problem{
	
	public:
	static map<string, Int> label_index_map;
	static vector<string> label_name_list;
	static Int D; // number of features
	static Int K; // number of classes
	//static Int* remap_indices;
	//static Int* rev_remap_indices;
	
	//number of samples
	Int N;
	vector<Instance*> data;

	Problem(char* fname){
		readData(fname);
	}
	
	void readData(char* fname){
		assert(false /* no implemented yet */);
	}

	Problem(ChainProblem* chain){
		construct(chain);
	}

	void construct(ChainProblem* chain){
		label_index_map = chain->label_index_map;
		label_name_list = chain->label_name_list;
		D = chain->D;
		K = chain->K;
		data.clear();
		for (auto it = chain->data.begin(); it != chain->data.end(); it++){
			Seq* seq = *it;
			//for each chain, reform it as a subgraph
			Instance* ins = new Instance();
			ins->features = seq->features;
			ins->labels = seq->labels;
			ins->T = seq->T;
			for (int i = 0; i < ins->T - 1; i++){
				ins->edges.push_back(make_pair(i, i+1));
			}
			ins->build_adj();
			data.push_back(ins);
		}
		
	}
	
	Problem(char* fname, string type){
		if (type == "chain"){	
			ChainProblem* chain = new ChainProblem(fname);
			construct(chain);
		} else {
			cerr << "unknown graph type: " << type << endl;
			exit(0);
		}
	}

};

map<string,Int> Problem::label_index_map;
vector<string> Problem::label_name_list;
Int Problem::D = -1;
Int Problem::K = -1;
//Int* Problem::remap_indices=NULL;
//Int* Problem::rev_remap_indices=NULL;


//parameters of a task
class Param{

	public:
	char* testFname;
	char* modelFname;
	int solver;
	Problem* prob;
	int max_iter;
	Float eta, rho;

	Param(){
		solver = 0;
		max_iter = 100;
		eta = 1.0;
		rho = 1.0;
		testFname = NULL;
		modelFname = NULL;
		prob = NULL;
	}
};

#endif
