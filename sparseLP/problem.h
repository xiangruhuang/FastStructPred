#ifndef PROBLEM_H
#define PROBLEM_H 

#include "util.h"
//#include "extra.h"
#include <cassert>

extern double prediction_time;

//parameters of a task
class Param{

	public:
		char* testFname;
		char* modelFname;
		int solver;
		int max_iter;
		Float eta, rho;
		string type; // problem type
		Float infea_tol; // tolerance of infeasibility
        Float grad_tol; // stopping condition for gradient
        Float nnz_tol; // threshold to shrink to zero

		Param(){
			solver = 0;
			max_iter = 1000;
			eta = 1.0;
			rho = 1.0;
			testFname = NULL;
			modelFname = NULL;
			type = "chain";
			infea_tol = 1e-3;
            grad_tol = 1e-3;
            nnz_tol = 1e-6;
		}
};

class ScoreVec{
	public:
		Float* c; // score vector: c[k1k2] = -v[k1k2/K][k1k2%K];
		pair<Float, int>* sorted_c; // sorted <score, index> vector; 
		pair<Float, int>** sorted_row; // sorted <score, index> vector of each row
		pair<Float, int>** sorted_col; // sorted <score, index> vector of each column
};

//each Instance is an isolated subgraph
class Instance{
	public:
		vector<Float*> node_score_vecs; 
		vector<ScoreVec*> edge_score_vecs;
		vector<vector<string>*> node_label_lists; 
		Labels labels; 
		int T; // number of nodes in this instance
		vector<pair<int, int>> edges; // edge list of this subgraph
		//vector<vector<Int>*> adj; // adjacency list of each node, only store edges towards nodes of larger id
		//void build_adj(){
		//	//assume edges is constructed, build adjacency list
		//	adj.clear();
		//	for (int i = 0; i < T; i++){
		//		adj.push_back(new vector<Int>());
		//	}
		//	for (vector<pair<Int, Int>>::iterator e = edges.begin(); e != edges.end(); e++){
		//		assert(e->first < e->second /* need i < j for each edge (i,j) */);
		//		adj[e->first]->push_back(e->second);
		//	}
		//}
};

class Problem{
	private:
		Problem(){
		};

	public:
		Param* param;
		vector<Instance*> data;
		Problem(Param* _param) : param(_param) {}
		virtual void construct_data(){
			cerr << "NEED to implement construct_data() for this problem!" << endl;
			assert(false);
		}
};

class CompleteGraphProblem : public Problem{
	public:
		map<string, int> label_index_map;
		vector<string> label_name_list;
		map<int, uni_factor*> node_index_map;
		int D, K, N;
		Float** w;
		Float** v;
		Float* c;
		void construct_data(){
			node_index_map.clear();
			label_index_map.clear();
			label_index_list.clear();
			ifstream fin(param->testFname);
			char* line = new char[LINE_LEN];

			Instance* ins = new Instance();
			Int d = -1;
			N = 0;
			fin.getline(line, LINE_LEN);
			string line_str(line);
			vector<string> tokens = split(line_str, " ");
			K = stoi(tokens[1]);
			int num_nodes = stoi(tokens[0]);
			for (int i = 0; i < num_nodes; i++){
				fin.getline(line, LINE_LEN);
				string line_str(line);
				tokens = split(line_str, " ");
				int node_ind = stoi(tokens[0]);
				Float* c = new Float[K];
				memset(c, 0.0, sizeof(Float)*K);
				for (vector<string>::iterator t = tokens.begin(); t != tokens.end(); t++){
					if (t == tokens.begin()){
						continue;
					}
					vector<string> label_val = split(*t, ":");
					map<string, int>::iterator it = label_index_map.find(label_val[0]);
					int index;
					if (it == label_index_map.end()){
						index = label_index_map.size();
						label_index_map.insert(make_pair(label_val[0], index));
					} else {
						index = it->second;
					}
					c[index] = stof(label_val[1]);
				}
				ins->node_score_vecs.push_back(c);
				
			}

			for (int k1 = 0; k1 < K; k1++){
				
			}
			while( !fin.eof() ){
				fin.getline(line, LINE_LEN);
				string line_str(line);

				if( line_str.length() < 2 && fin.eof() ){
					if(ins->labels.size()>0){
						ins->T = ins->labels.size();
						data.push_back(ins);
						N++;
					}
					break;
				}else if( line_str.length() < 2 ){
					ins->T = ins->labels.size();
					data.push_back(ins);
					N++;
					ins = new Instance();
					continue;
				}
				vector<string> tokens = split(line_str, " ");
				//get label index
				Int lab_ind;
				map<string,Int>::iterator it;
				if(  (it=label_index_map.find(tokens[0])) == label_index_map.end() ){
					lab_ind = label_index_map.size();
					label_index_map.insert(make_pair(tokens[0],lab_ind));
				}else
					lab_ind = it->second;

				SparseVec* x = new SparseVec();
				for(Int i=1;i<tokens.size();i++){
					vector<string> kv = split(tokens[i],":");
					Int ind = atoi(kv[0].c_str());
					Float val = atof(kv[1].c_str());
					x->push_back(make_pair(ind,val));

					if( ind > d )
						d = ind;
				}

				//compute c = -W^T x
				Float* c = new Float[K];
				memset(c, 0.0, sizeof(Float)*K);
				for (SparseVec::iterator it = x->begin(); it != x->end(); it++){
					Float* wj = w[it->first];
					for (int k = 0; k < K; k++)
						c[k] -= wj[k]*it->second;
				}
				x->clear();


				ins->node_score_vecs.push_back(c);
				ins->labels.push_back(lab_ind);
				int len = ins->labels.size();
				if (len >= 2){
					ins->edge_score_vecs.push_back(sv);
					ins->edges.push_back(make_pair(len-2, len-1));
				}
			}
			fin.close();

			d += 1; //bias
			if( D < d )
				D = d;

			for(Int i=0;i<data.size();i++)
				data[i]->T = data[i]->labels.size();

			label_name_list.resize(label_index_map.size());

			for(map<string,Int>::iterator it=label_index_map.begin(); it!=label_index_map.end(); it++){
				label_name_list[it->second] = it->first;
			}

			//propagate address of label name list to all nodes
			for (vector<Instance*>::iterator it_data = data.begin(); it_data != data.end(); it_data++){
				Instance* ins = *it_data;
				for (int t = 0; t < ins->T; t++){
					ins->node_label_lists.push_back(&(label_name_list));
				}
			}

			K = label_index_map.size();

			delete[] line;
			
		}
}

class ChainProblem : public Problem{

	public:
		map<string,Int> label_index_map;
		vector<string> label_name_list;
		int D;
		int K;
		int N; // number of instances
		//from model
		Float** w; // w[:D][:K]
		Float** v;
		Float* c; // c[k1*K+k2] = -v[k1][k2];
		pair<Float, int>* sorted_c; // sorted <score, index> vector; 
		pair<Float, int>** sorted_row; // sorted <score, index> vector of each row
		pair<Float, int>** sorted_col; // sorted <score, index> vector of each column
		vector<pair<int, Float>>* sparse_v;
		ScoreVec* sv;

		ChainProblem(Param* _param) : Problem(_param) {}

		~ChainProblem(){
			label_index_map.clear();
			label_name_list.clear();
			for (int j = 0; j < D; j++)
				delete[] w[j];
			delete[] w;
			for (int k = 0; k < K; k++)
				delete[] v[k];
			delete[] v;
			for (vector<Instance*>::iterator it_d = data.begin(); it_d != data.end(); it_d++){
				Instance* ins = *it_d;
				//score vectors are not duplicated, thus they can be freed
				for (vector<Float*>::iterator it_node = ins->node_score_vecs.begin(); it_node != ins->node_score_vecs.end(); it_node++){
					delete[] *it_node;
				}
				ins->node_score_vecs.clear();
				//but edge score vectors and node label lists are not duplicated
				ins->edge_score_vecs.clear();
				ins->node_label_lists.clear();
				ins->labels.clear();
				ins->edges.clear();
			}
			for (int k = 0; k < K; k++)
				sparse_v[k].clear();
			delete[] sparse_v;
			delete[] c;
			delete[] sorted_c;
			for (int k = 0; k < K; k++){
				delete[] sorted_row[k];
				delete[] sorted_col[k];
			}
			delete[] sorted_row;
			delete[] sorted_col;
		}

		void readTestData(char* fname){
			ifstream fin(fname);
			char* line = new char[LINE_LEN];

			Instance* ins = new Instance();
			Int d = -1;
			N = 0;
			while( !fin.eof() ){
				fin.getline(line, LINE_LEN);
				string line_str(line);

				if( line_str.length() < 2 && fin.eof() ){
					if(ins->labels.size()>0){
						ins->T = ins->labels.size();
						data.push_back(ins);
						N++;
					}
					break;
				}else if( line_str.length() < 2 ){
					ins->T = ins->labels.size();
					data.push_back(ins);
					N++;
					ins = new Instance();
					continue;
				}
				vector<string> tokens = split(line_str, " ");
				//get label index
				Int lab_ind;
				map<string,Int>::iterator it;
				if(  (it=label_index_map.find(tokens[0])) == label_index_map.end() ){
					lab_ind = label_index_map.size();
					label_index_map.insert(make_pair(tokens[0],lab_ind));
				}else
					lab_ind = it->second;

				SparseVec* x = new SparseVec();
				for(Int i=1;i<tokens.size();i++){
					vector<string> kv = split(tokens[i],":");
					Int ind = atoi(kv[0].c_str());
					Float val = atof(kv[1].c_str());
					x->push_back(make_pair(ind,val));

					if( ind > d )
						d = ind;
				}

				//compute c = -W^T x
				Float* c = new Float[K];
				memset(c, 0.0, sizeof(Float)*K);
				for (SparseVec::iterator it = x->begin(); it != x->end(); it++){
					Float* wj = w[it->first];
					for (int k = 0; k < K; k++)
						c[k] -= wj[k]*it->second;
				}
				x->clear();


				ins->node_score_vecs.push_back(c);
				ins->labels.push_back(lab_ind);
				int len = ins->labels.size();
				if (len >= 2){
					ins->edge_score_vecs.push_back(sv);
					ins->edges.push_back(make_pair(len-2, len-1));
				}
			}
			fin.close();

			d += 1; //bias
			if( D < d )
				D = d;

			for(Int i=0;i<data.size();i++)
				data[i]->T = data[i]->labels.size();

			label_name_list.resize(label_index_map.size());

			for(map<string,Int>::iterator it=label_index_map.begin(); it!=label_index_map.end(); it++){
				label_name_list[it->second] = it->first;
			}

			//propagate address of label name list to all nodes
			for (vector<Instance*>::iterator it_data = data.begin(); it_data != data.end(); it_data++){
				Instance* ins = *it_data;
				for (int t = 0; t < ins->T; t++){
					ins->node_label_lists.push_back(&(label_name_list));
				}
			}

			K = label_index_map.size();

			delete[] line;
		}

		void readModel(char* fname){
			ifstream fin(fname);
			char* line = new char[LINE_LEN];
			//first line, get K
			fin.getline(line, LINE_LEN);
			string line_str(line);
			vector<string> tokens = split(line_str, "=");
			K = stoi(tokens[1]);

			//second line, get label_name_list
			fin.getline(line, LINE_LEN);
			line_str = string(line);
			tokens = split(line_str, " ");

			//token[0] is 'label', means nothing
			for (Int i = 1; i < tokens.size(); i++){
				label_name_list.push_back(tokens[i]);
				label_index_map.insert(make_pair(tokens[i], (i-1)));
			}

			//third line, get D
			fin.getline(line, LINE_LEN);
			line_str = string(line);
			tokens = split(line_str, "=");
			D = stoi(tokens[1]);
			//skip fourth line
			fin.getline(line, LINE_LEN);
			//next D lines: read w
			w = new Float*[D];
			for (Int j = 0; j < D; j++){
				w[j] = new Float[K];
				memset(w[j], 0, sizeof(Float)*K);
				fin.getline(line, LINE_LEN);
				line_str = string(line);
				tokens = split(line_str, " ");
				Float* wj = w[j];
				for (vector<string>::iterator it = tokens.begin(); it != tokens.end(); it++){
					vector<string> k_w = split(*it, ":");
					Int k = stoi(k_w[0]);
					wj[k] = stof(k_w[1]);
				}
			}

			//skip next line
			fin.getline(line, LINE_LEN);

			//next K lines: read v
			v = new Float*[K];
			for (Int k1 = 0; k1 < K; k1++){
				v[k1] = new Float[K];
				memset(v[k1], 0, sizeof(Float)*K);
				fin.getline(line, LINE_LEN);
				line_str = string(line);
				tokens = split(line_str, " ");
				Float* v_k1 = v[k1];
				for (vector<string>::iterator it = tokens.begin(); it != tokens.end(); it++){
					vector<string> k2_v = split(*it, ":");
					Int k2 = stoi(k2_v[0]);
					v_k1[k2] = stof(k2_v[1]);
				}
			}
			sparse_v = new vector<pair<int, Float>>[K];
			for (int k1 = 0; k1 < K; k1++){
				sparse_v[k1].clear();
				for (int k2 = 0; k2 < K; k2++){
					if (fabs(v[k1][k2]) > 1e-12){
						sparse_v[k1].push_back(make_pair(k2, v[k1][k2]));
					}
				}
			}

			c = new Float[K*K];
			for (int kk = 0; kk < K*K; kk++)
				c[kk] = -v[kk/K][kk%K];

			if (param->solver == 1){
				//extra sorting time is counted
				prediction_time -= omp_get_wtime();
			}
			//sort c as well as each row and column in increasing order
			int K1 = K;
			int K2 = K;
			sorted_row = new pair<Float, int>*[K1];
			for (int k1 = 0; k1 < K1; k1++){
				sorted_row[k1] = new pair<Float, int>[K2];
			}
			sorted_col = new pair<Float, int>*[K2];
			for (int k2 = 0; k2 < K2; k2++){
				sorted_col[k2] = new pair<Float, int>[K1];
			}
			sorted_c = new pair<Float, int>[K1*K2];
			for (int k1 = 0; k1 < K1; k1++){
				int offset = k1*K2;
				pair<Float, int>* sorted_row_k1 = sorted_row[k1];
				for (int k2 = 0; k2 < K2; k2++){
					Float val = c[offset+k2];
					sorted_c[offset+k2] = make_pair(val, offset+k2);
					sorted_row_k1[k2] = make_pair(val, k2);
					sorted_col[k2][k1] = make_pair(val, k1);
				}
			}
			for (int k1 = 0; k1 < K1; k1++){
				sort(sorted_row[k1], sorted_row[k1]+K2, less<pair<Float, int>>());
			}
			for (int k2 = 0; k2 < K2; k2++){
				sort(sorted_col[k2], sorted_col[k2]+K1, less<pair<Float, int>>());
			}
			sort(sorted_c, sorted_c+K1*K2, less<pair<Float, int>>());
			//store them in ScoreVec
			sv = new ScoreVec();
			sv->c = c;
			sv->sorted_c = sorted_c;
			sv->sorted_row = sorted_row;
			sv->sorted_col = sorted_col;

			if (param->solver == 1){
				//extra sorting time is counted
				prediction_time += omp_get_wtime();
			}
		}
		void construct_data(){
			//read model first, construct label name list and label index map
			readModel(param->modelFname);
			//read data from test file, labels will be determined by label index map from model
			readTestData(param->testFname);
		}


		/*void construct(ChainProblem* chain){
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
		}*/

};

//map<string,Int> Problem::label_index_map;
//vector<string> Problem::label_name_list;
//Int Problem::D = -1;
//Int Problem::K = -1;
//Int* Problem::remap_indices=NULL;
//Int* Problem::rev_remap_indices=NULL;



#endif
