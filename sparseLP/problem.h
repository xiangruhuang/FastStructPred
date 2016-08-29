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
		string problem_type; // problem type
		Float infea_tol; // tolerance of infeasibility
		Float grad_tol; // stopping condition for gradient
		Float nnz_tol; // threshold to shrink to zero
		bool MultiLabel;

		Param(){
			solver = 0;
			max_iter = 1000;
			eta = 1.0;
			rho = 1.0;
			testFname = NULL;
			modelFname = NULL;
			problem_type = "NULL";
			infea_tol = 1e-4;
			grad_tol = 1e-4;
			nnz_tol = 1e-8;
			MultiLabel = false;
		}
};

class ScoreVec{
	public:
		Float* c; // score vector: c[k1k2] = -v[k1k2/K][k1k2%K];
		pair<Float, int>* sorted_c; // sorted <score, index> vector; 
		pair<Float, int>** sorted_row; // sorted <score, index> vector of each row
		pair<Float, int>** sorted_col; // sorted <score, index> vector of each column
		int K1, K2;
		ScoreVec(Float* _c, int _K1, int _K2){
			//sort c as well as each row and column in increasing order
			c = _c;
			K1 = _K1;
			K2 = _K2;
			internal_sort();
		}
		ScoreVec(int _K1, int _K2, Float* _c){
			c = _c;
			K1 = _K1;
			K2 = _K2;
		}

		void internal_sort(){
			if (sorted_row != NULL && sorted_col != NULL && sorted_c != NULL){
				return;
			}
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
		}

		/*void normalize(Float smallest, Float largest){
			assert(normalized == false);
			normalized = true;
			Float width = max(largest - smallest, 1e-12);
			for (int i = 0; i < K1*K2; i++){
				c[i] = (c[i]-smallest)/width;
				sorted_c[i].first = (sorted_c[i].first - smallest)/width;
				assert(c[i] >= 0 && c[i] <= 1);
				assert(sorted_c[i].first >= 0 && sorted_c[i].first <= 1);
			}
			for (int k1 = 0; k1 < K1; k1++){
				for (int k2 = 0; k2 < K2; k2++){
					sorted_row[k1][k2].first = (sorted_row[k1][k2].first - smallest)/width;
					sorted_col[k2][k1].first = (sorted_col[k2][k1].first - smallest)/width;
					assert(sorted_row[k1][k2].first >= 0 && sorted_row[k1][k2].first <= 1);
					assert(sorted_col[k2][k1].first >= 0 && sorted_col[k2][k1].first <= 1);
				}
			}
		}*/

		~ScoreVec(){
			delete[] c;
			delete[] sorted_c;
			for (int i = 0; i < K1; i++){
				delete[] sorted_row[i];
			}
			delete[] sorted_row;
			for (int i = 0; i < K2; i++){
				delete[] sorted_col[i];
			}
			delete[] sorted_col;
		}
	private: bool normalized = false;

};

//each Instance is an isolated subgraph
class Instance{
	public:
		vector<pair<Float*, int>> table_in_memory;
		Float smallest, largest; // new = (original-smallest)/(largest-smallest); new \in [0,1] guaranteed; original = new * (largest - smallest) + smallest;
		vector<Float*> node_score_vecs; 
		vector<ScoreVec*> edge_score_vecs;
		vector<vector<string>*> node_label_lists; 
		Labels labels; // true labels
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
		void normalize(){
			Float width = max(largest - smallest, 1e-12);
			for (vector<pair<Float*, int>>::iterator it = table_in_memory.begin(); it != table_in_memory.end(); it++){
				Float* c = it->first;
				int L = it->second;
				for (int i = 0; i < L; i++){
					c[i] = (c[i] - smallest)/width;
				}
			}
		}
};

class Problem{
	public:
		Problem(){
		};

		Param* param;
		vector<Instance*> data;
		Problem(Param* _param) : param(_param) {}
		virtual void construct_data(){
			cerr << "NEED to implement construct_data() for this problem!" << endl;
			assert(false);
		}
};

inline void readLine(ifstream& fin, char* line){
	fin.getline(line, LINE_LEN);
	while (!fin.eof() && strlen(line) == 0){
		fin.getline(line, LINE_LEN);
	}
}


class UAIProblem : public Problem{
	public:
		UAIProblem(Param* _param) : Problem(_param){}
		~UAIProblem(){

		}

		void construct_data(){
			char* fname = param->testFname;
			char* line = new char[LINE_LEN];
			cerr << "reading uai file " << fname << endl;
			ifstream fin(fname);
			//skip first line
			readLine(fin, line);
			//only one instance
			Instance* ins = new Instance();
			readLine(fin, line);
			ins->T = atoi(line);
			int T = ins->T;
			//cerr << "T=" << T << endl;
			readLine(fin, line);
			string line_str(line);
			vector<string> tokens = split(line_str, " ");
			assert(tokens.size() == T);
			for (int i = 0; i < T; i++){
				vector<string>* list_i = new vector<string>();
				list_i->clear();
				int K = stoi(tokens[i]);
				for (int k = 0; k < K; k++){
					list_i->push_back(to_string(k));
				}
				ins->node_label_lists.push_back(list_i);
				Float* c_i = new Float[K];
				memset(c_i, 0.0, sizeof(Float)*K);
				ins->node_score_vecs.push_back(c_i);
				ins->labels.push_back(0);
			}

			// number of factors
			readLine(fin, line);
			int F = atoi(line);
			//cerr << "F=" << F << endl;
			vector<string> tables;
			map<pair<int, int>, Float*> table_map;
			for (int f = 0; f < F; f++){
				readLine(fin, line);
				string line_str(line);
				tables.push_back(line_str);
				tokens = split(line_str, " ");
				int d = stoi(tokens[0]);
				if (d == 1){
					//this is a node
					//nothing to do
				} else {
					assert(d == 2);
					//this is an edge
					int l = stoi(tokens[1]), r = stoi(tokens[2]);
					//cerr << "l=" << l << ", r=" << r << endl;
					pair<int, int> e = make_pair(l, r); //l*T + r;
					int K1 = ins->node_label_lists[l]->size();
					int K2 = ins->node_label_lists[r]->size();
					//cerr << "K1=" << K1 << ", K2=" << K2 << endl;
					if (table_map.find(e) == table_map.end()){
						ins->edges.push_back(e);
						Float* c_e = new Float[K1*K2];
						memset(c_e, 0.0, sizeof(Float)*K1*K2);
						table_map.insert(make_pair(e, c_e));
						ScoreVec* sv = new ScoreVec(K1, K2, c_e);
						ins->edge_score_vecs.push_back(sv);
					}
				}
			}

			//cerr << "done with preamble" << endl;

			// read tables
			for (int f = 0; f < F; f++){
				//cerr << f << "/" << F << endl;
				tokens = split(tables[f], " ");
				int d = stoi(tokens[0]);
				if (d == 1){
					//this is a node
					int i = stoi(tokens[1]);
					Float* c_i = ins->node_score_vecs[i];
					//readLine(fin, line);
					int K;
					fin >> K; //= atoi(line);
					assert(K == ins->node_label_lists[i]->size());
					/*readLine(fin, line);
					  string line_str(line);
					  tokens = split(line_str, " ");
					  assert(tokens.size() == K);*/
					for (int k = 0; k < K; k++){
						Float val; // = stof(tokens[k]);
						fin >> val;
						c_i[k] += -log(val);
					}
				} else {
					assert(d == 2);
					//this is an edge
					int l = stoi(tokens[1]), r = stoi(tokens[2]);
					pair<int, int> e = make_pair(l, r);
					int K1 = ins->node_label_lists[l]->size(), K2 = ins->node_label_lists[r]->size();
					map<pair<int, int>, Float*>::iterator it = table_map.find(e);
					assert(it != table_map.end());
					Float* c_e = it->second;
					table_map.insert(make_pair(e, c_e));
					int KK;
					fin >> KK;
					assert(KK = K1 * K2);
					for (int k1 = 0; k1 < K1; k1++){
						for (int k2 = 0; k2 < K2; k2++){
							Float val;
							fin >> val;
							c_e[k1*K2+k2] += -log(val);
						}
					}
				}
			}
			//cerr << "done reading uai file" << endl;			

			//sort each ScoreVec after all values read
			for (int i = 0; i < ins->edges.size(); i++){
				ScoreVec* sv = ins->edge_score_vecs[i];
				sv->internal_sort();
			}
			//cerr << "done sorting" << endl;			

			data.push_back(ins);
		}
};

class LOGUAIProblem : public UAIProblem{
	public:
		LOGUAIProblem(Param* _param) : UAIProblem(_param){}
		~LOGUAIProblem(){

		}

		void construct_data(){
			char* fname = param->testFname;
			char* line = new char[LINE_LEN];
			cerr << "reading loguai file " << fname << endl;
			ifstream fin(fname);
			//skip first line
			readLine(fin, line);
			//only one instance
			Instance* ins = new Instance();
			readLine(fin, line);
			ins->T = atoi(line);
			int T = ins->T;
			//cerr << "T=" << T << endl;
			readLine(fin, line);
			string line_str(line);
			vector<string> tokens = split(line_str, " ");
			assert(tokens.size() == T);
			for (int i = 0; i < T; i++){
				vector<string>* list_i = new vector<string>();
				list_i->clear();
				int K = stoi(tokens[i]);
				for (int k = 0; k < K; k++){
					list_i->push_back(to_string(k));
				}
				ins->node_label_lists.push_back(list_i);
				Float* c_i = new Float[K];
				memset(c_i, 0.0, sizeof(Float)*K);
				ins->node_score_vecs.push_back(c_i);
				ins->labels.push_back(0);
			}

			// number of factors
			readLine(fin, line);
			line_str = string(line);
			//vector<string> F_total = split(line_str, " ");
			
			int F=stoi(line_str);// total;
			//F = stoi(F_total[0]);
			//total = stoi(F_total[1]);
			cerr << "F=" << F; // << ", total=" << total << endl;
			vector<string> tables;
			//map<pair<int, int>, Float*> table_map;
			for (int f = 0; f < F; f++){
				readLine(fin, line);
				string line_str(line);
				tables.push_back(line_str);
				/*tokens = split(line_str, " ");
				  int d = stoi(tokens[0]);
				  if (d == 1){
				//this is a node
				//nothing to do
				} else {
				assert(d == 2);
				//this is an edge
				int l = stoi(tokens[1]), r = stoi(tokens[2]);
				//cerr << "l=" << l << ", r=" << r << endl;
				pair<int, int> e = make_pair(l, r); //l*T + r;
				int K1 = ins->node_label_lists[l]->size();
				int K2 = ins->node_label_lists[r]->size();
				//cerr << "K1=" << K1 << ", K2=" << K2 << endl;
				if (table_map.find(e) == table_map.end()){
				ins->edges.push_back(e);
				Float* c_e = new Float[K1*K2];
				memset(c_e, 0.0, sizeof(Float)*K1*K2);
				table_map.insert(make_pair(e, c_e));
				ScoreVec* sv = new ScoreVec(K1, K2, c_e);
				ins->edge_score_vecs.push_back(sv);
				}
				}*/
			}

			//cerr << "done with preamble" << endl;

			// read tables
			ins->table_in_memory.clear(); 
			for (int f = 0; f < F; f++){
				//cerr << f << "/" << F << endl;
				int L = -1;
				fin >> L;
				Float* c = new Float[L];
				ins->table_in_memory.push_back(make_pair(c, L));
				for (int l = 0; l < L; l++){
					Float val;
					fin >> val;
					c[l] = -val;
					if (c[l] < ins->smallest){
						ins->smallest = c[l];
					} 	
					if (c[l] > ins->largest){
						ins->largest = c[l];
					}
				}
				ScoreVec* sv = NULL;
				vector<string> all_tokens = split(tables[f], ";");
				for (int tok = 0; tok < all_tokens.size(); tok++){
					tokens = split(all_tokens[tok], " ");
					//cerr << all_tokens[tok] << endl;
					int d = stoi(tokens[0]);
					if (d == 1){
						//this is a node
						int i = stoi(tokens[1]);
						ins->node_score_vecs[i] = c;
						//readLine(fin, line);
						//int K;
						//fin >> K; //= atoi(line);
						assert(L == ins->node_label_lists[i]->size());
						/*readLine(fin, line);
						  string line_str(line);
						  tokens = split(line_str, " ");
						  assert(tokens.size() == K);*/
						/*for (int k = 0; k < K; k++){
						  Float val; // = stof(tokens[k]);
						  fin >> val;
						  c_i[k] += -log(val);
						  }*/
					} else {
						assert(d == 2);
						//this is an edge
						int l = stoi(tokens[1]), r = stoi(tokens[2]);
						pair<int, int> e = make_pair(l, r);
						ins->edges.push_back(e);
						int K1 = ins->node_label_lists[l]->size(), K2 = ins->node_label_lists[r]->size();
						assert(L == K1 * K2);
						if (sv == NULL){
							sv = new ScoreVec(K1, K2, c);
						}
						ins->edge_score_vecs.push_back(sv);
					}
				}
			}
			cerr << "done reading uai file" << endl;			

			ins->normalize();
			//sort each ScoreVec after all values read
			for (int i = 0; i < ins->edges.size(); i++){
				ScoreVec* sv = ins->edge_score_vecs[i];
				sv->internal_sort();
			}
			cerr << "done normalizing" << endl;			
			data.push_back(ins);
		}
};

class MultiLabelProblem : public Problem{
	public:
		map<string, int> label_index_map;
		vector<string> label_name_list;
		map<string, int> node_index_map;
		int K, D, N;
		ScoreVec* sv;
		Float** w;
		MultiLabelProblem(Param* _param) : Problem(_param){}
		~MultiLabelProblem(){
			label_index_map.clear();
			label_name_list.clear();
			node_index_map.clear();
			for (int j = 0; j < D; j++)
				delete[] w[j];
			delete[] w;
		}
		void readTestData(char* fname){
			ifstream fin(fname);
			char* line = new char[LINE_LEN];

			Int d = -1;
			N = 0;
			while( !fin.eof() ){
				fin.getline(line, LINE_LEN);
				string line_str(line);

				size_t found = line_str.find("  ");
				while (found != string::npos){
					line_str = line_str.replace(found, 2, " ");
					found = line_str.find("  ");
				}
				found = line_str.find(", ");
				while (found != string::npos){
					line_str = line_str.replace(found, 2, ",");
					found = line_str.find(", ");
				}
				found = line_str.find(" ,");
				while (found != string::npos){
					line_str = line_str.replace(found, 2, ",");
					found = line_str.find(" ,");
				}
				if( line_str.length() < 2 && fin.eof() ){
					continue;
				}

				Instance* ins = new Instance();
				vector<string> tokens = split(line_str, " ");
				//get label index
				Int lab_ind;
				map<string,Int>::iterator it;
				vector<string> labels = split(tokens[0], ",");
				for (int i = 0; i < labels.size(); i++){
					string label = labels[i];
					if(  (it=label_index_map.find(label)) == label_index_map.end() ){
						lab_ind = label_index_map.size();
						label_index_map.insert(make_pair(label,lab_ind));
					}else
						lab_ind = it->second;
					ins->labels.push_back(lab_ind);
				}

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
				int len = ins->labels.size();
				ins->edge_score_vecs.push_back(sv);
				//write ins to uai format

				data.push_back(ins);
			}
			fin.close();

			d += 1; //bias
			if( D < d )
				D = d;

			label_name_list.resize(label_index_map.size());


			for(map<string,Int>::iterator it=label_index_map.begin(); it!=label_index_map.end(); it++){
				label_name_list[it->second] = it->first;
			}

			K = label_index_map.size();

			for(Int i=0;i<data.size();i++){
				data[i]->T = K; 
			}

			//propagate address of label name list to all nodes
			for (vector<Instance*>::iterator it_data = data.begin(); it_data != data.end(); it_data++){
				Instance* ins = *it_data;
				ins->node_label_lists.push_back(&(label_name_list));
			}

			/*ofstream fout("../../data/multilabel/rcv1_regions.uai");
			  Instance* ins = data[0];
			  fout << "MARKOV" << endl;
			  int T = ins->T;
			  fout << T << endl;
			  for (int t = 0; t < T; t++){
			  if (t != 0)
			  fout << " ";
			  fout << 2;
			  }
			  fout << endl;
			  fout << (T + K*(K-1)/2) << endl;
			  for (int t = 0; t < T; t++){
			  fout << "1 " << t << endl;
			  }
			  for (int k1 = 0; k1 < K; k1++){
			  for (int k2 = k1+1; k2 < K; k2++){
			  fout << "2 " << k1 << " " << k2 << endl; 
			  }
			  }
			  for (int i = 0; i < T; i++){
			  fout << 2 << endl;
			  Float c_i = ins->node_score_vecs[0][i];
			  fout << setprecision(10) << 1.0 << " " << exp(-c_i) << endl;
			  fout << endl;
			  }
			  for (int k1 = 0; k1 < K; k1++){
			  for (int k2 = k1+1; k2 < K; k2++){
			  fout << 4 << endl;
			  Float c_e = ins->edge_score_vecs[0]->c[k1*K+k2];
			  fout << 1.0 << " " << 1.0 << endl;
			  fout << setprecision(10) << 1.0 << " " << exp(-c_e) << endl;
			  fout << endl;
			  }
			  }

			  fout.close();*/

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
			Float** v = new Float*[K];
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

			Float* c = new Float[K*K];
			for (int kk = 0; kk < K*K; kk++)
				c[kk] = -v[kk/K][kk%K];

			if (param->solver == 2){
				//extra sorting time is counted
				prediction_time -= omp_get_wtime();
			}
			//store them in ScoreVec
			sv = new ScoreVec(c, K, K);

			if (param->solver == 2){
				//extra sorting time is counted
				prediction_time += omp_get_wtime();
			}
		}

		void construct_data(){
			readModel(param->modelFname);
			readTestData(param->testFname);
		}
};

class CompleteGraphProblem : public Problem{
	public:
		map<string, int> label_index_map;
		vector<string> label_name_list;
		map<string, int> node_index_map;
		int K;
		ScoreVec* sv;
		CompleteGraphProblem(Param* _param) : Problem(_param) {}
		~CompleteGraphProblem(){
			label_index_map.clear();
			label_name_list.clear();
			node_index_map.clear();
		}
		void construct_data(){
			cerr << "constructing from " << param->testFname << " ";
			node_index_map.clear();
			label_index_map.clear();
			label_name_list.clear();
			ifstream fin(param->testFname);
			char* line = new char[LINE_LEN];

			Instance* ins = new Instance();// only one instance in this task
			readLine(fin, line);
			string line_str(line);
			vector<string> tokens = split(line_str, " ");
			K = stoi(tokens[1]);
			ins->T = stoi(tokens[0]);
			for (int i = 0; i < ins->T; i++){
				readLine(fin, line);
				string line_str(line);
				tokens = split(line_str, " ");
				node_index_map.insert(make_pair(tokens[0], i));
				Float* c = new Float[K];
				memset(c, 0.0, sizeof(Float)*K);
				for (vector<string>::iterator t = tokens.begin() + 1; t != tokens.end(); t++){
					vector<string> label_val = split(*t, ":");
					map<string, int>::iterator it = label_index_map.find(label_val[0]);
					int index;
					if (it == label_index_map.end()){
						index = label_index_map.size();
						label_index_map.insert(make_pair(label_val[0], index));
					} else {
						index = it->second;
					}
					c[index] = (Float)(-stod(label_val[1]));
				}
				ins->node_score_vecs.push_back(c);
				ins->labels.push_back(0); // prediction is disabled for this task
			}
			//cerr << "#node=" << ins->T << endl;

			//skip this line
			readLine(fin, line);
			Float* c = new Float[K*K];
			memset(c, 0.0, sizeof(Float)*K*K);
			for (int i = 0; i < K; i++){
				readLine(fin, line);
				string line_str(line);
				vector<string> tokens = split(line_str, " ");
				map<string, int>::iterator it = label_index_map.find(tokens[0]);
				assert(it != label_index_map.end());
				int k1 = it->second;
				assert(k1 >= 0 && k1 < K);
				for (int j = 1; j < tokens.size(); j++){
					vector<string> label_val = split(tokens[j], ":");
					Float val = (Float)stod(label_val[1]);
					map<string, int>::iterator it = label_index_map.find(label_val[0]);
					assert(it != label_index_map.end());
					int k2 = it->second;
					assert(k2 >= 0 && k2 < K);
					c[k1*K+k2] = -val;
				}
			}

			sv = new ScoreVec(c, K, K);
			while (!fin.eof()){
				readLine(fin, line);
				if (fin.eof() || strlen(line) == 0) continue;
				string line_str(line);
				vector<string> tokens = split(line_str, " ");
				cerr << line_str << endl;
				map<string, int>::iterator i = node_index_map.find(tokens[0]), j = node_index_map.find(tokens[1]);
				assert(i != node_index_map.end() && j != node_index_map.end());
				ins->edges.push_back(make_pair(i->second,j->second));
				ins->edge_score_vecs.push_back(sv);
			}

			//cerr << "#edges=" << ins->edges.size() << endl;

			data.push_back(ins);

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

			ofstream fout("../../data/107network/107network.uai");
			fout << "MARKOV" << endl;
			int T = ins->T;
			fout << T << endl;
			for (int t = 0; t < T; t++){
				if (t != 0)
					fout << " ";
				fout << K ;
			}
			fout << endl;
			fout << (T + ins->edges.size()) << endl;
			for (int t = 0; t < T; t++){
				fout << "1 " << t << endl;
			}
			for (int t = 0; t < ins->edges.size(); t++){
				fout << "2 " << ins->edges[t].first << " " << ins->edges[t].second << endl;
			}
			for (int i = 0; i < T; i++){
				fout << K << endl;
				Float* c_i = ins->node_score_vecs[i];
				for (int k = 0; k < K; k++){
					if (k != 0)
						fout << " ";
					fout << exp(-c_i[k]);
				}
				fout << endl;
				fout << endl;
			}
			for (int e = 0; e < ins->edges.size(); e++){
				cerr << e << "/" << ins->edges.size() << endl;
				int l = ins->edges[e].first, r = ins->edges[e].second;
				Float* c_e = ins->edge_score_vecs[e]->c;
				int K1 = ins->node_label_lists[l]->size(), K2 = ins->node_label_lists[r]->size();
				fout << K1*K2 << endl;
				for (int k1 = 0; k1 < K1; k1++){
					for (int k2 = 0; k2 < K2; k2++){
						if (k2 != 0)
							fout << " ";
						fout << exp(-c_e[k1*K2+k2]);
					}
					fout << endl;
				}
				fout << endl;
			}

			fout.close();

			delete[] line;
			cerr << "done" << endl;
		}
};

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
			//store them in ScoreVec
			sv = new ScoreVec(c, K, K);

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
