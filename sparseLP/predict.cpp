#include "problem.h"
#include "factor.h"
#include <time.h>
double prediction_time = 0.0;

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
		int label = -1;
		double s = 0.0;
		int K = ins->node_label_lists[node_count]->size();
		for (int k = 0; k < K; k++){
			s += node->y[k];
		}
		Float r = s*((Float)rand()/RAND_MAX);
		for (int k = 0; k < K; k++){
			if (r <= node->y[k]){
				label = k;
				break;
			} else {
				r -= node->y[k];
			}
		}
		assert(label != -1 /* prediction should be valid */);
		//if (ins->node_label_lists[node_count]->at(ins->labels[node_count]) == model->label_name_list->at(label)){
		if (ins->labels[node_count] == label){
			hit++;
		}
	}
	return (Float)hit/ins->T;
}

void struct_predict(Problem* prob, Param* param){
	vector<uni_factor*> nodes; 
	vector<bi_factor*> edges;
	Float hit = 0.0;
	Float N = 0.0;
	double uni_search_time = 0.0;
	double uni_subsolve_time = 0.0;
	double bi_search_time = 0.0;
	double bi_subsolve_time = 0.0;
	double maintain_time = 0.0;
	double construct_time = 0.0;
	for (vector<Instance*>::iterator it_ins = prob->data.begin(); it_ins != prob->data.end(); it_ins++){
		Instance* ins = *it_ins;
		construct_time -= omp_get_wtime();
		construct_factor(ins, param, nodes, edges);
		construct_time += omp_get_wtime();
		int iter = 0;
		int max_iter = param->max_iter;
		Float score = 0.0;
		Float p_inf = 0.0;
		Float val = 0.0;
		Float act_on_node = 0.0, act_on_edge = 0.0;
		Float ever_act_on_node = 0.0;
        Float nnz_msg = 0.0;
        Float acc = 0.0;
        while (iter < max_iter){
			score = 0.0;
			p_inf = 0.0;
			val = 0.0;
			act_on_node = 0.0;
			act_on_edge = 0.0;
            ever_act_on_node = 0.0;
			nnz_msg = 0.0;
            for (vector<uni_factor*>::iterator it_node = nodes.begin(); it_node != nodes.end(); it_node++){
				uni_factor* node = *it_node;

				uni_search_time -= omp_get_wtime();
				node->search();
				uni_search_time += omp_get_wtime();

				uni_subsolve_time -= omp_get_wtime();
				node->subsolve();
				uni_subsolve_time += omp_get_wtime();

                prediction_time += omp_get_wtime();
				score += node->score();
				val += node->func_val();
				act_on_node += node->act_set.size();
				ever_act_on_node += node->ever_act_set.size();
				//node->display();
                prediction_time -= omp_get_wtime();
			}
			act_on_node /= nodes.size();
            ever_act_on_node /= nodes.size();

			for (vector<bi_factor*>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++){
				bi_factor* edge = *it_edge;

				bi_search_time -= omp_get_wtime();
				edge->search();
				bi_search_time += omp_get_wtime();

				bi_subsolve_time -= omp_get_wtime();
				edge->subsolve();
				bi_subsolve_time += omp_get_wtime();

                prediction_time += omp_get_wtime();
				score += edge->score();
				val += edge->func_val();
				act_on_edge += edge->act_set.size();
				p_inf += edge->infea();
                nnz_msg += edge->nnz_msg();
				//edge->display();
                prediction_time -= omp_get_wtime();
			}
			act_on_edge /= edges.size();
            nnz_msg /= edges.size();

			for (vector<bi_factor*>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++){
				bi_factor* edge = *it_edge;
				maintain_time -= omp_get_wtime();
				edge->update_multipliers();
				maintain_time += omp_get_wtime();
			}
 
			if (p_inf < 1e-6)
				break;
			
			iter++;
		}
        acc = compute_acc(ins, nodes);
        cerr << "iter=" << iter 
            //<< ", T=" << edges.size()+1
            << ", avg_act_node=" << act_on_node
            << ", avg_act_edge=" << act_on_edge
            << ", avg_ever_act_node=" << ever_act_on_node
            << ", nnz_msg=" << nnz_msg
            << ", acc=" << acc
            << ", score=" << score 
            << ", val=" << val
            << ", infea=" << p_inf
            << endl;
		hit += acc*ins->T;
		N += ins->T;
	}
	cerr << "uni_search=" << uni_search_time
		<< ", uni_subsolve=" << uni_subsolve_time
		<< ", bi_search=" << bi_search_time
		<< ", bi_subsolve=" << bi_subsolve_time 
		<< ", maintain=" << maintain_time 
		<< ", construct=" << construct_time 
		<< ", Acc=" << hit/N << endl;
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
		
		//int temp_hit = hit;
		for(Int t=0;t<ins->T;t++){
			if( pred[t] == ins->labels[t] )
				hit++;
		}

		//cerr << (double)(hit-temp_hit)/(seq->T) << endl;

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
		cerr << "prob.D=" << ((ChainProblem*)prob)->D << endl;
		cerr << "prob.K=" << ((ChainProblem*)prob)->K << endl;
	}
	
	cerr << "prob.N=" << prob->data.size() << endl;
	
	cerr << "param.rho=" << param->rho << endl;
	cerr << "param.eta=" << param->eta << endl;

	prediction_time = -omp_get_wtime();
	if (param->solver == 0){
		cerr << "Acc=" << Viterbi_predict((ChainProblem*)prob, param) << endl;
	} else {
		//cerr << "Acc=" << compute_acc_sparseLP(model, prob) << endl;
		struct_predict(prob, param);
	}
	prediction_time += omp_get_wtime();
	cerr << "prediction time=" << prediction_time << endl;
	return 0;	
}
