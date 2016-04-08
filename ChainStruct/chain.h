#ifndef CHAIN_H
#define CHAIN_H
#include "util.h"
#include <cassert>
class Seq{
	public:
	vector<SparseVec*> features;
	Labels labels;
       	Int T;
};

class ChainProblem{
	
	public:
	static map<string,Int> label_index_map;
	static vector<string> label_name_list;
	static Int D;
	static Int K;
	static Int* remap_indices;
	static Int* rev_remap_indices;
	vector<Seq*> data;
	Int N;

	ChainProblem(char* fname){
		readData(fname);
	}
	
	void readData(char* fname){
		ifstream fin(fname);
		char* line = new char[LINE_LEN];
		
		Seq* seq = new Seq();
		Int d = -1;
		N = 0;
		while( !fin.eof() ){

			fin.getline(line, LINE_LEN);
			string line_str(line);
			
			if( line_str.length() < 2 && fin.eof() ){
				if(seq->labels.size()>0)
					data.push_back(seq);
				break;
			}else if( line_str.length() < 2 ){
				data.push_back(seq);
				seq = new Seq();
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
			seq->features.push_back(x);
			seq->labels.push_back(lab_ind);
			N++;
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
		
		K = label_index_map.size();
		
		delete[] line;
	}
	void label_random_remap(){
		srand(time(NULL));
		if (remap_indices == NULL || rev_remap_indices == NULL){
			remap_indices = new Int[K];
			for (Int k = 0; k < K; k++)
				remap_indices[k] = k;
			random_shuffle(remap_indices, remap_indices+K);
			rev_remap_indices = new Int[K];
			for (Int k = 0; k < K; k++)
				rev_remap_indices[remap_indices[k]] = k;
			label_index_map.clear();
			for (Int ind = 0; ind < K; ind++){
				label_index_map.insert(make_pair(label_name_list[ind], remap_indices[ind]));
			}
			label_name_list.clear();
			label_name_list.resize(label_index_map.size());
			for(map<string,Int>::iterator it=label_index_map.begin(); it!=label_index_map.end(); it++){
				label_name_list[it->second] = it->first;
			}
		}
		for (int i = 0; i < data.size(); i++){
			Seq* seq = data[i];
			for (int t = 0; t < seq->T; t++){
				Int yi = seq->labels[t];
				seq->labels[t] = remap_indices[yi];
			}
		}
	}

	void print_freq(){
		Int* freq = new Int[K];
		memset(freq, 0, sizeof(Int)*K);
		for (int i = 0; i < data.size(); i++){
			Seq* seq = data[i];
			for (int t = 0; t < seq->T; t++){
				Int yi = seq->labels[t];
				freq[yi]++;
			}
		}
		sort(freq, freq + K, greater<Int>());
		for (int k = 0; k < K; k++)
			cout << freq[k] << " ";
		cout << endl;
	}

	void simple_shuffle(){
		
		for (int i = 0; i < data.size(); i++){
			Seq* seq = data[i];
			for (int t = 0; t < seq->T; t++){
				Int yi = seq->labels[t];
				seq->labels[t] = (yi+13)%K;
			}
		}
	}
};

map<string,Int> ChainProblem::label_index_map;
vector<string> ChainProblem::label_name_list;
Int ChainProblem::D = -1;
Int ChainProblem::K;
Int* ChainProblem::remap_indices=NULL;
Int* ChainProblem::rev_remap_indices=NULL;

class Model{
	
	public:
	Model(char* fname){
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
		label_name_list = new vector<string>();
		label_index_map = new map<string, Int>();
		//token[0] is 'label', means nothing
		for (Int i = 1; i < tokens.size(); i++){
			label_name_list->push_back(tokens[i]);
			label_index_map->insert(make_pair(tokens[i], (i-1)));
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
		sparse_v = new vector<pair<Int, Float>>[K];
		for (Int k1 = 0; k1 < K; k1++){
			sparse_v[k1].clear();
			for (Int k2 = 0; k2 < K; k2++){
				if (fabs(v[k1][k2]) > 1e-12){
					sparse_v[k1].push_back(make_pair(k2, v[k1][k2]));
				}
			}
		}
	}

	Model(Float** _w, Float** _v, ChainProblem* prob){
		D = prob->D;
		K = prob->K;
		label_name_list = &(prob->label_name_list);
		label_index_map = &(prob->label_index_map);
		w = _w;
		v = _v;
		sparse_v = new vector<pair<Int, Float>>[K];
		for (Int k1 = 0; k1 < K; k1++){
			sparse_v[k1].clear();
			for (Int k2 = 0; k2 < K; k2++){
				if (fabs(v[k1][k2]) > 1e-12){
					sparse_v[k1].push_back(make_pair(k2, v[k1][k2]));
				}
			}
		}
	}
	
	~Model(){
		for (Int k = 0; k < K; k++){
		       	sparse_v[k].clear();
		}
		delete[] sparse_v;
	}

	Float** w;
	Float** v;
	Int D;
	Int K;
	vector<string>* label_name_list;
	map<string,Int>* label_index_map;
	vector<pair<Int, Float>>* sparse_v;

	void writeModel( char* fname ){

		ofstream fout(fname);
		fout << "nr_class K=" << K << endl;
		fout << "label ";
		for(vector<string>::iterator it=label_name_list->begin();
				it!=label_name_list->end(); it++)
			fout << *it << " ";
		fout << endl;
		fout << "nr_feature D=" << D << endl;
		fout << "unigram w, format: D lines; line j contains (k, w) forall w[j][k] neq 0" << endl;
		for (int j = 0; j < D; j++){
			Float* wj = w[j];
			bool flag = false;
			for (int k = 0; k < K; k++){
				if (fabs(wj[k]) < 1e-12)
					continue;
				if (flag)
					fout << " ";
				else
					flag = true;
				fout << k << ":" << wj[k];
			}
			fout << endl;
		}

		fout << "bigram v, format: K lines; line k1 contains (k2, v) forall v[k1][k2] neq 0" << endl;
		for (int k1 = 0; k1 < K; k1++){
			Float* v_k1 = v[k1];
			bool flag = false;
			for (int k2 = 0; k2 < K; k2++){
				if (fabs(v_k1[k2]) < 1e-12)
					continue;
				if (flag)
					fout << " ";
				else 
					flag = true;
				fout << k2 << ":" << v_k1[k2];
			}
			fout << endl;
		}
		fout.close();
	}

	Float calcAcc_Viterbi(ChainProblem* prob){
		vector<Seq*>* data = &(prob->data);
		Int nSeq = data->size();
		Int N = 0;
		Int hit=0;
		for(Int n=0;n<nSeq;n++){
			
			Seq* seq = data->at(n);
			N += seq->T;
			//compute prediction
			Int* pred = new Int[seq->T];
			Float** max_sum = new Float*[seq->T];
			Int** argmax_sum = new Int*[seq->T];
			for(Int t=0; t<seq->T; t++){
				max_sum[t] = new Float[K];
				argmax_sum[t] = new Int[K];
				for(Int k=0;k<K;k++)
					max_sum[t][k] = -1e300;
			}
			////Viterbi t=0
			SparseVec* xi = seq->features[0];
			for(Int k=0;k<K;k++)
				max_sum[0][k] = 0.0;
			for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
				Float* wj = w[it->first];
				for(Int k=0;k<K;k++)
					max_sum[0][k] += wj[k]*it->second;
			}
			////Viterbi t=1...T-1
			for(Int t=1; t<seq->T; t++){
				//passing message from t-1 to t
				for(Int k1=0;k1<K;k1++){
					Float tmp = max_sum[t-1][k1];
					Float cand_val;
					/*for(Int k2=0;k2<K;k2++){
						 cand_val = tmp + v[k1][k2];
						 if( cand_val > max_sum[t][k2] ){
							max_sum[t][k2] = cand_val;
							argmax_sum[t][k2] = k1;
						 }
					}*/
					for (vector<pair<Int, Float>>::iterator it = sparse_v[k1].begin(); it != sparse_v[k1].end(); it++){
						Int k2 = it->first; 
						cand_val = tmp + it->second;
						if( cand_val > max_sum[t][k2] ){
							max_sum[t][k2] = cand_val;
							argmax_sum[t][k2] = k1;
						}	
					}
				}
				//adding unigram factor
				SparseVec* xi = seq->features[t];
				for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++)
					for(Int k2=0;k2<K;k2++)
						max_sum[t][k2] += w[it->first][k2] * it->second;
			}
			////Viterbi traceback
			pred[seq->T-1] = argmax( max_sum[seq->T-1], K );
			for(Int t=seq->T-1; t>=1; t--){
				pred[t-1] = argmax_sum[t][ pred[t] ];
			}
			
			//compute accuracy
			for(Int t=0;t<seq->T;t++){
				assert(label_name_list == &(prob->label_name_list));
				//if( label_name_list->at(pred[t]) == prob->label_name_list[seq->labels[t]] )
				if( pred[t] == seq->labels[t] )
					hit++;
			}
			
			for(Int t=0; t<seq->T; t++){
				delete[] max_sum[t];
				delete[] argmax_sum[t];
			}
			delete[] max_sum;
			delete[] argmax_sum;
			delete[] pred;
		}
		Float acc = (Float)hit/N;
		
		return acc;
	}
};

class Param{

	public:
	char* trainFname;
	char* heldoutFname;
	char* modelFname;
	Float C;
	ChainProblem* prob;
	ChainProblem* heldout_prob;
	
	int solver;
	int max_iter;
	Float eta; //Augmented-Lagrangian parameter
	bool using_brute_force;
	int split_up_rate;
	int write_model_period;
	int early_terminate;
	Float admm_step_size;
	bool do_subSolve;
	Int heldout_period;
	Param(){
		solver = 0;
		C = 1.0;
		max_iter =100000;
		eta = 0.1;
		heldout_prob = NULL;
		using_brute_force = false;
		split_up_rate = 1;
		write_model_period = 0;
		early_terminate = 3;
		admm_step_size = 1.0;
		do_subSolve = true;
		heldout_period = 10;
	}
};


void chain_logDecode(double* nodePot, double* edgePot, int num_vars, int num_states, Labels* y_decode){
	
	double** max_marg = new double*[num_vars];
	int** argmax = new int*[num_vars];
	for(int i=0;i<num_vars;i++){
		max_marg[i] = new double[num_states];
		argmax[i] = new int[num_states];
		for(int j=0;j<num_states;j++){
			max_marg[i][j] = -1e300;
			argmax[i][j] = -1;
		}
	}
	
	//initialize
	for(int i=0;i<num_states;i++)
		max_marg[0][i] = nodePot[0*num_states+i];
	//forward pass
	for(int t=1;t<num_vars;t++){
		for(int i=0;i<num_states;i++){
			int ind = i*num_states;
			for(int j=0;j<num_states;j++){
				double new_val =  max_marg[t-1][i] + edgePot[ind+j];
				if( new_val > max_marg[t][j] ){
					max_marg[t][j] = new_val;
					argmax[t][j] = i;
				}
			}
		}
		for(int i=0;i<num_states;i++)
			max_marg[t][i] += nodePot[t*num_states+i];
	}
	//backward pass to find the ystar	
	y_decode->clear();
	y_decode->resize(num_vars);
	
	double max_val = -1e300;
	for(int i=0;i<num_states;i++){
		if( max_marg[num_vars-1][i] > max_val ){
			max_val = max_marg[num_vars-1][i];
			(*y_decode)[num_vars-1] = i;
		}
	}
	for(int t=num_vars-2;t>=0;t--){
		(*y_decode)[t] = argmax[t+1][ (*y_decode)[t+1] ];
	}
	
	for(int t=0;t<num_vars;t++)
		delete[] max_marg[t];
	delete[] max_marg;

	for(int t=0;t<num_vars;t++)
		delete[] argmax[t];
	delete[] argmax;
}

void chain_oracle(Param* param, Model* model, Seq* seq, Labels* y, Labels* ystar){
	
	Int T = seq->T;
	Int D = param->prob->D;
	Int K = param->prob->K;
	
	double* theta_unary = new double[T*K];
	double* theta_pair = new double[K*K];
	Float** w = model->w;
	Float** v = model->v;

	for(int t=0; t<T; t++){
		SparseVec* x_t = seq->features[t];
		int tK = t*K;
		
		for(int k=0; k<K; k++)
			theta_unary[tK+k] = 0.0;
		for( SparseVec::iterator it=x_t->begin(); it!=x_t->end(); it++){
			Int j = it->first;
			double xt_j = it->second;
			for(int k=0; k<K; k++){
				theta_unary[tK + k] += w[j][k] * xt_j;
			}
		}
	}
	int bigram_offset = D*K;
	for(int k1k2=0; k1k2<K*K; k1k2++){
		Int k1= k1k2 / K, k2 = k1k2 % K;
		theta_pair[k1k2] = v[k1][k2];
	}
	
	//loss augmented
	if( y != NULL ){
		for(int t=0; t<T; t++){
			for(int k=0; k<K; k++){
				if( k != y->at(t) )
					//theta_unary[ t*K +k ] += 1.0/T;
					theta_unary[ t*K +k ] += 1.0;
			}
		}
	}
	
	// decode
	chain_logDecode(theta_unary, theta_pair, T, K, ystar);
	
	delete[] theta_unary;
	delete[] theta_pair;
}

double primal_obj( Param* param, int total, int subsample,  Model* model){
	
	vector<Seq*>* data = &(param->prob->data);

	Int N = param->prob->N;
	Int D = param->prob->D;
	Int K = param->prob->K;
	Float** w = model->w;
	Float** v = model->v;

	Labels* ystar = new Labels();
	double loss_term = 0.0;
	Int m = subsample;
	if (m > total)
		m = total;
	vector<Int> indices;
	for(Int i = 0; i < total; i++){
		indices.push_back(i);
	}
	random_shuffle(indices.begin(), indices.end());
	for(Int mm = 0; mm < m; mm++){
		Int i = indices[mm];	
		Seq* seq = data->at(i);
		Labels* labels = &(seq->labels);
		chain_oracle(param, model, seq, labels, ystar);
		Int T = labels->size();

		for (Int t = 0; t < T; t++){
			Int ystar_t = ystar->at(t), yn_t = labels->at(t);
			SparseVec* x_t = seq->features[t];
			for (SparseVec::iterator it_x = x_t->begin(); it_x != x_t->end(); it_x++){
				Int j = it_x->first;
				Int xij = it_x->second;
				loss_term += (w[j][ystar_t] - w[j][yn_t]) * xij;
			}
		}
		for (Int t = 0; t < T - 1; t++){
			loss_term += v[labels->at(t)][labels->at(t+1)];
			loss_term -= v[ystar->at(t)][ystar->at(t+1)];
		}

		for (Int t = 0; t < T; t++){
			if (labels->at(t) != ystar->at(t)){
				//loss_term += 1.0/T;
				loss_term += 1.0;
			}
		}
	}

	loss_term *= ((Float)total*1.0/m);
	
	double reg_term = 0.0;
	//lambda = 1.0/C
	for(int j = 0; j < D; j++){
		for (Int k = 0; k < K; k++){
			double wbar_val = w[j][k];
			reg_term += wbar_val*wbar_val;
		}
	}
	for(int k1 = 0; k1 < K; k1++){
		for (Int k2 = 0; k2 < K; k2++){
			double wbar_val = v[k1][k2];
			reg_term += wbar_val*wbar_val;
		}
	}
	reg_term /= (2*param->C);
	loss_term /= param->C;
	delete ystar;
	
	return reg_term + loss_term;
}

#endif 
