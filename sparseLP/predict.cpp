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

	Float hit = 0;
	//for this instance

	//compute hits
	int node_count = 0;
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
	for (vector<Instance*>::iterator it_ins = prob->data.begin(); it_ins != prob->data.end(); it_ins++, n++){
        Instance* ins = *it_ins;
		stats->construct_time -= get_current_time();
		construct_factor(ins, param, nodes, edges);
        int* node_indices = new int[nodes.size()];
        for (int i = 0; i < nodes.size(); i++)
            node_indices[i] = i;
        int* edge_indices = new int[edges.size()];
        for (int i = 0; i < edges.size(); i++)
            edge_indices[i] = i;
		stats->construct_time += get_current_time();
		int iter = 0;
		int max_iter = param->max_iter;
		Float score, p_inf, d_inf, acc, nnz_msg;
		//Float val = 0.0;
		stats->clear();
        int countdown = 0;

        vector<factor*> factor_seq;
        for (int i = 0; i < ins->T; i++){
            factor_seq.push_back(nodes[i]);
            if (i < ins->T - 1)
                factor_seq.push_back(edges[i]);
        }

		nnz_msg = 0.0;
		while (iter < max_iter){
			score = 0.0;
			//val = 0.0;
            /*random_shuffle(node_indices, node_indices+nodes.size());
			for (int n = 0; n < nodes.size(); n++){
				uni_factor* node = nodes[node_indices[n]];

				stats->uni_search_time -= get_current_time();
				node->search();
				stats->uni_search_time += get_current_time();

				stats->uni_subsolve_time -= get_current_time();
				node->subsolve();
				stats->uni_subsolve_time += get_current_time();

			}

            random_shuffle(edge_indices, edge_indices+edges.size());
			for (int e = 0; e < edges.size(); e++){
				bi_factor* edge = edges[edge_indices[e]];
                
				stats->bi_search_time -= get_current_time();
				edge->search();
				stats->bi_search_time += get_current_time();

				stats->bi_subsolve_time -= get_current_time();
				edge->subsolve();
				stats->bi_subsolve_time += get_current_time();

			}*/

            for (vector<factor*>::iterator f = factor_seq.begin(); f != factor_seq.end(); f++){
                //cerr << "search" << endl;
                (*f)->search();
                //cerr << "subsolve" << endl;
                (*f)->subsolve();
                //cerr << "end" << endl;
            }

			for (vector<bi_factor*>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++){
				bi_factor* edge = *it_edge;
				edge->update_multipliers();
			}

            //compute primal inf & dual inf
            
            //cerr << "==========================" << endl;
            d_inf = 0.0; int dinf_index = -1; int dinf_factor_index = -1;
            int node_count = 0;
			for (vector<uni_factor*>::iterator it_node = nodes.begin(); it_node != nodes.end(); it_node++, node_count++){
				uni_factor* node = *it_node;
                Float gmax_node = node->dual_inf();
                if (gmax_node > d_inf){
                    d_inf = gmax_node;
                    dinf_index = node->dinf_index;
                    dinf_factor_index = node_count;
                    //cerr << setprecision(10) << "node" << node_count << ":" << d_inf << ", act_size=" << node->act_set.size() << ", k=" << dinf_index << ", y=" << node->y[dinf_index] << ", grad=" << node->grad[dinf_index] << endl;
                }
				prediction_time += get_current_time();
				score += node->score();
				//val += node->func_val();
				stats->uni_act_size += node->act_set.size();
				stats->uni_ever_act_size += node->ever_act_set.size();
				stats->num_uni++;
				prediction_time -= get_current_time();
            }
           
            int edge_count = 0;
            p_inf = 0.0;
			for (vector<bi_factor*>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++, edge_count++){
				bi_factor* edge = *it_edge;
                Float gmax_edge = edge->dual_inf();
                if (gmax_edge > d_inf){
                    d_inf = gmax_edge;
                    dinf_index = edge->dinf_index;
                    dinf_factor_index = edge_count + nodes.size();
                    //cerr << "edge" << edge_count << " " << d_inf << endl;
                }
				Float inf = edge->infea();
                if (inf > p_inf)
                    p_inf = inf;
				prediction_time += get_current_time();
				score += edge->score();
				//val += edge->func_val();
                stats->bi_act_size += edge->act_set.size();
				stats->num_bi++;
				nnz_msg += edge->nnz_msg();
				prediction_time -= get_current_time();
            }
            Float g = 0.0;  
            if (dinf_factor_index != -1){
                if (dinf_factor_index >= nodes.size()){
                    edge_count = dinf_factor_index - nodes.size();
                    bi_factor* edge = edges[edge_count];
                    int k1k2 = dinf_index;
                    int k1 = k1k2 / edge->K2, k2 = k1k2 % edge->K2;
                    g = -(edge->c[dinf_index] + edge->msgl[k1] + edge->msgr[k2]);
                } else {
                    g = nodes[dinf_factor_index]->grad[dinf_index];
                }
            }
            //cerr << ", d_inf=" << d_inf << ", dinf_factor=" << dinf_factor_index << ", k=" << dinf_index << ", grad=" << g << endl;
            //cerr << "==========================" << endl;
            //cerr << ", grad[997]=" << nodes[1]->grad[997] << ", grad[429]=" << nodes[1]->grad[429] << endl;
            //if (nodes[1]->searched_index != -1){
                //cerr << ", searched_index=" << nodes[1]->searched_index << ", gmax=" << nodes[1]->grad[nodes[1]->searched_index] << endl;
            //}
            //cerr << ", msgl[429]=" << edges[1]->msgl[429] << ", msgl[997]=" << edges[1]->msgl[997];
            //cerr << ", msgr[1163]=" << edges[1]->msgr[1163];
            //nodes[1]->display();
			if (p_inf < param->infea_tol && d_inf < param->grad_tol){
				countdown++;
            }
            if (countdown >= 10){
                break;
            }
            /*if (iter > 10){
                if (!edges[1]->inside[997*3039+1163]){
                    edges[1]->act_set.push_back(make_pair(997*3039+1163, 0.0));
                    edges[1]->inside[997*3039+1163]=true;
                }
            }*/
			iter++;
		}
		//nnz_msg /= edges.size()*(iter+1);

		prediction_time += get_current_time();
		acc = compute_acc(ins, nodes);
		prediction_time -= get_current_time();

		cerr << "@" << n
			<< ": iter=" << iter
            << ", T=" << ins->T
			<< ", acc=" << acc
			<< ", score=" << score 
			<< ", infea=" << p_inf
            << ", dinf=" << d_inf
            << ", avg_nnz_msg=" << nnz_msg / (iter+1) / edges.size();
        
        Float ever_act_size = 0;
        for (int i = 0; i < nodes.size(); i++)
            ever_act_size += nodes[i]->ever_act_set.size();
        ever_act_size /= nodes.size();
        cerr << ", ever_act_size=" << ever_act_size;

        cerr << ", uni_search=" << stats->uni_search_time
            << ", uni_subsolve=" << stats->uni_subsolve_time
            << ", bi_search=" << stats->bi_search_time
            << ", bi_subsolve=" << stats->bi_subsolve_time 
            << ", maintain=" << stats->maintain_time 
            << ", construct=" << stats->construct_time << endl; 
        Float avg_act_size = 0.0;
        Float tscore = 0.0;
        for (int i = 0; i < nodes.size(); i++){
            avg_act_size += nodes[i]->act_set.size();
            //cerr << "true_label=" << ins->labels[i] << endl;
            //tscore += nodes[i]->score();
            //cerr << "tscore=" << tscore << ", c[k]=" << nodes[i]->c[nodes[i]->act_set[0]]<< " ";
            //nodes[i]->display();
            //if (i < nodes.size()-1){
            //    tscore += edges[i]->score();
            //    cerr << "tscore=" << tscore << " ,score=" << edges[i]->score();
            //    edges[i]->display();
            //}
        }
        cerr << ", avg_act_size=" << avg_act_size / nodes.size();
		stats->display();

        /*
        int y0, y1, y2;
        y0=1570; y1=997; y2 = 1163;
        cerr << "(y0,y1,y2)=(" << y0 << "," << y1 << "," << y2 << ")" << ": c0=" << nodes[0]->c[y0] << ", v0=" << edges[0]->c[y0*3039 + y1] << ", c1=" << nodes[1]->c[y1] << ", v1=" << edges[0]->c[y1*3039 + y2] <<endl;
        
        y0=1570; y1=429; y2 = 1163;
        cerr << "(y0,y1,y2)=(" << y0 << "," << y1 << "," << y2 << ")" << ": c0=" << nodes[0]->c[y0] << ", v0=" << edges[0]->c[y0*3039 + y1] << ", c1=" << nodes[1]->c[y1] << ", v1=" << edges[0]->c[y1*3039 + y2] <<endl;
        */

		hit += acc*ins->T;
		N += ins->T;
		cerr << ", Acc=" << hit/N << endl;
		cerr << endl;

        delete[] node_indices;
        delete[] edge_indices;
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
            Float* max_sum_t = max_sum[t];
            Float* c = ins->edge_score_vecs[t-1]->c;
            int offset = 0;
			for(Int k1=0;k1<K;k1++){
				Float tmp = max_sum[t-1][k1];
				Float cand_val;
				for (int k2 = 0; k2 < K; k2++){
                    cand_val = tmp - c[offset+k2];
                    if( cand_val > max_sum_t[k2] ){
						max_sum_t[k2] = cand_val;
						argmax_sum[t][k2] = k1;
					}
				}
                offset += K;
			}
			//adding unigram factor
            Float* score_at_t = ins->node_score_vecs[t];
			for(Int k2=0;k2<K;k2++)
				max_sum[t][k2] -= score_at_t[k2];
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
            //cerr << "t=" << t << ", pred=" << pred[t] << ", score=" << max_sum[t][pred[t]] << endl;
		}

        cerr << "@" << n << ": acc=" << (double)(hit-temp_hit)/(ins->T) << " --- " << (Float)hit/N << ", score=" << max_sum[ins->T-1][pred[ins->T-1]]<< endl;

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
