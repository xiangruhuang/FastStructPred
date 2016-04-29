#include "problem.h"
#include "factor.h"
#include <time.h>
#include "LPsparse.h"

double prediction_time = 0.0;
extern Stats* stats;

void exit_with_help(){
	cerr << "Usage: ./train (options) [testfile] [model]" << endl;
	cerr << "options:" << endl;
	cerr << "-s solver: (default 0)" << endl;
	cerr << "	0 -- Viterbi(chain)" << endl;
	cerr << "	1 -- sparseLP" << endl;
	cerr << "-e eta: GDMM step size" << endl;
	cerr << "-o rho: coefficient/weight of message" << endl;
	cerr << "-m max_iter: max number of iterations" << endl;
	exit(0);
}

void parse_cmd_line(int argc, char** argv, Param* param){

	int i;
	for(i=1;i<argc;i++){
		if( argv[i][0] != '-' )
			break;
		if( ++i >= argc )
			exit_with_help();

		switch(argv[i-1][1]){
			
			case 's': param->solver = atoi(argv[i]);
				  break;
			case 'e': param->eta = atof(argv[i]);
				  break;
			case 'o': param->rho = atof(argv[i]);
				  break;
			case 'm': param->max_iter = atoi(argv[i]);
				  break;
			case 't': param->infea_tol = atof(argv[i]);
				  break;
			default:
				  cerr << "unknown option: -" << argv[i-1][1] << endl;
				  exit(0);
		}
	}

	if(i>=argc)
		exit_with_help();

	param->testFname = argv[i];
	i++;
	if( i<argc )
		param->modelFname = argv[i];
	else{
		param->modelFname = new char[FNAME_LEN];
		strcpy(param->modelFname,"model");
	}
}

inline void construct_factor(Instance* ins, Param* param, vector<uni_factor*>& nodes, vector<bi_factor*>& edges){
	nodes.clear();
	edges.clear();

	//construct uni_factors
	int node_count = 0;
	for (int t = 0; t < ins->T; t++){
		uni_factor* node = new uni_factor(ins->node_label_lists[t]->size(), ins->node_score_vecs[t], param);
		nodes.push_back(node);
	}

	//construct bi_factors
	for (int e = 0; e < ins->edges.size(); e++){
		int e_l = ins->edges[e].first, e_r = ins->edges[e].second;
		ScoreVec* sv = ins->edge_score_vecs[e];
		uni_factor* l = nodes[e_l];
		uni_factor* r = nodes[e_r];
		bi_factor* edge = new bi_factor(l, r, sv, param);
		edges.push_back(edge);
	}
}

inline Float compute_acc(Instance* ins, vector<uni_factor*> nodes){

	int node_count = 0;
	int hit = 0;
	int N = 0;
	//for this instance

	//compute hits
	for (vector<uni_factor*>::iterator it_node = nodes.begin(); it_node != nodes.end(); it_node++, node_count++){	
		uni_factor* node = *it_node;
		//Rounding
		hit+=node->y[ins->labels[node_count]];
	}
	return (Float)hit/ins->T;
}

double struct_predict(Problem* prob, Param* param){
	vector<uni_factor*> nodes; 
	vector<bi_factor*> edges;
	Float hit = 0.0;
	Float N = 0.0;
	int n = 0;
	stats = new Stats();
	for (vector<Instance*>::iterator it_ins = prob->data.begin(); it_ins != prob->data.end(); it_ins++){
		Instance* ins = *it_ins;
		stats->construct_time -= get_current_time();
		construct_factor(ins, param, nodes, edges);
		stats->construct_time += get_current_time();
		int iter = 0;
		int max_iter = param->max_iter;
		Float score = 0.0;
		Float p_inf = 0.0;
		//Float val = 0.0;
		Float acc = 0.0;
		stats->clear();
		while (iter < max_iter){
			score = 0.0;
			p_inf = 0.0;
			//val = 0.0;
			//nnz_msg = 0.0;
			for (vector<uni_factor*>::iterator it_node = nodes.begin(); it_node != nodes.end(); it_node++){
				uni_factor* node = *it_node;

				stats->uni_search_time -= get_current_time();
				node->search();
				stats->uni_search_time += get_current_time();

				stats->uni_subsolve_time -= get_current_time();
				node->subsolve();
				stats->uni_subsolve_time += get_current_time();

				prediction_time += get_current_time();
				score += node->score();
				//val += node->func_val();
				stats->uni_act_size += node->act_set.size();
				stats->uni_ever_act_size += node->ever_act_set.size();
				stats->num_uni++;
				//node->display();
				prediction_time -= get_current_time();
			}

			for (vector<bi_factor*>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++){
				bi_factor* edge = *it_edge;

				stats->bi_search_time -= get_current_time();
				edge->search();
				stats->bi_search_time += get_current_time();

				stats->bi_subsolve_time -= get_current_time();
				edge->subsolve();
				stats->bi_subsolve_time += get_current_time();

				prediction_time += get_current_time();
				score += edge->score();
				//val += edge->func_val();
				stats->bi_act_size += edge->act_set.size();
				stats->num_bi++;
				p_inf += edge->infea();
				//nnz_msg += edge->nnz_msg();
				//edge->display();
				prediction_time -= get_current_time();
			}

			for (vector<bi_factor*>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++){
				bi_factor* edge = *it_edge;
				stats->maintain_time -= get_current_time();
				edge->update_multipliers();
				stats->maintain_time += get_current_time();
			}

			if (p_inf < param->infea_tol)
				break;

			iter++;
		}
		n++;
		//nnz_msg /= edges.size()*(iter+1);

		prediction_time += get_current_time();
		acc = compute_acc(ins, nodes);
		prediction_time -= get_current_time();

		cerr << "@" << n 
			<< ": iter=" << iter 
			<< ", acc=" << acc
			<< ", score=" << score 
			<< ", infea=" << p_inf;

		stats->display();

		hit += acc*ins->T;
		N += ins->T;
		cerr << ", Acc=" << hit/N << endl;
		cerr << endl;
	}
	cerr << "uni_search=" << stats->uni_search_time
		<< ", uni_subsolve=" << stats->uni_subsolve_time
		<< ", bi_search=" << stats->bi_search_time
		<< ", bi_subsolve=" << stats->bi_subsolve_time 
		<< ", maintain=" << stats->maintain_time 
		<< ", construct=" << stats->construct_time << endl; 
	return (double)hit/(double)N;
}

Float Viterbi_predict(ChainProblem* prob, Param* param){
	vector<Instance*>* data = &(prob->data);
	int N = 0;
	int hit=0;
	int K = prob->K;
	vector<pair<int, Float>>* sparse_v = prob->sparse_v;
	for(Int n=0;n<prob->N;n++){
		Instance* ins = data->at(n);
		N += ins->T;
		//compute prediction
		Int* pred = new Int[ins->T];
		Float** max_sum = new Float*[ins->T];
		Int** argmax_sum = new Int*[ins->T];
		for(Int t=0; t<ins->T; t++){
			max_sum[t] = new Float[K];
			argmax_sum[t] = new Int[K];
			for(Int k=0;k<K;k++)
				max_sum[t][k] = -1e300;
		}
		////Viterbi t=0
		for(Int k=0;k<K;k++)
			max_sum[0][k] = 0.0;
		for(Int k=0;k<K;k++)
			max_sum[0][k] -= ins->node_score_vecs[0][k];
		////Viterbi t=1...T-1
		for(Int t=1; t<ins->T; t++){
			//passing message from t-1 to t
			for(Int k1=0;k1<K;k1++){
				Float tmp = max_sum[t-1][k1];
				Float cand_val;
				for (vector<pair<int, Float>>::iterator it = sparse_v[k1].begin(); it != sparse_v[k1].end(); it++){
					Int k2 = it->first;
					cand_val = tmp + it->second;
					if( cand_val > max_sum[t][k2] ){
						max_sum[t][k2] = cand_val;
						argmax_sum[t][k2] = k1;
					}
				}
			}
			//adding unigram factor
			for(Int k2=0;k2<K;k2++)
				max_sum[t][k2] -= ins->node_score_vecs[t][k2];
		}
		////Viterbi traceback
		pred[ins->T-1] = argmax( max_sum[ins->T-1], K );
		for(Int t=ins->T-1; t>=1; t--){
			pred[t-1] = argmax_sum[t][ pred[t] ];
		}

		//compute accuracy
		
		int temp_hit = hit;
		for(Int t=0;t<ins->T;t++){
			if( pred[t] == ins->labels[t] )
				hit++;
		}

		cerr << (double)(hit-temp_hit)/(ins->T) << endl;

		for(Int t=0; t<ins->T; t++){
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

int main(int argc, char** argv){
	if (argc < 2){
		exit_with_help();
	}

    srand(time(NULL));
	Param* param = new Param();
	parse_cmd_line(argc, argv, param);
	
	Problem* prob;
	if (param->type == "chain"){
		prob = new ChainProblem(param);
		prob->construct_data();
        int D = ((ChainProblem*)prob)->D; 
        int K = ((ChainProblem*)prob)->K; 
		cerr << "prob.D=" << D << endl;
		cerr << "prob.K=" << K << endl;
        	cerr << "prob.nnz(v)=" << nnz( ((ChainProblem*)prob)->v, K, K, 1e-12 ) << endl; 
	}
	
	cerr << "prob.N=" << prob->data.size() << endl;
	
	cerr << "param.rho=" << param->rho << endl;
	cerr << "param.eta=" << param->eta << endl;

	prediction_time = -get_current_time();
	if (param->solver == 0){
		cerr << "Acc=" << Viterbi_predict((ChainProblem*)prob, param) << endl;
	} 
	if (param->solver == 1){
		cerr << "Acc=" << compute_acc_sparseLP(prob) << endl;
	}
	if (param->solver == 2){
		cerr << "Acc=" << struct_predict(prob, param) << endl;
	}
	prediction_time += get_current_time();
	cerr << "prediction time=" << prediction_time << endl;
	return 0;	
}
