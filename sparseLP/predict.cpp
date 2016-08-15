#include "problem.h"
#include "factor.h"
#include <time.h>
#include "LPsparse.h"
#include "multifactor.h"

double prediction_time = 0.0;
extern Stats* stats;
bool debug = false;

void exit_with_help(){
	cerr << "Usage: ./predict (options) [testfile] [model]" << endl;
	cerr << "options:" << endl;
	cerr << "-s solver: (default 0)" << endl;
	cerr << "	0 -- Viterbi(chain)" << endl;
	cerr << "	1 -- sparseLP" << endl;
	cerr << "       2 -- GDMM" << endl;
	cerr << "-p problem_type: " << endl;
	cerr << "   chain -- sequence labeling problem" << endl;
	cerr << "   network -- network matching problem" << endl;
	cerr << "   uai -- uai format problem" << endl;
	cerr << "-e eta: GDMM step size" << endl;
	cerr << "-o rho: coefficient/weight of message" << endl;
	cerr << "-m max_iter: max number of iterations" << endl;
	exit(0);
}

void parse_cmd_line(int argc, char** argv, Param* param){

	int i;
	vector<string> args;
	for (i = 1; i < argc; i++){
		string arg(argv[i]);
		//cerr << "arg[i]:" << arg << "|" << endl;
		args.push_back(arg);
	}
	for(i=0;i<args.size();i++){
		string arg = args[i];
		if (arg == "-debug"){
			debug = true;
			continue;
		}
		if( arg[0] != '-' )
			break;

		if( ++i >= args.size() )
			exit_with_help();

		string arg2 = args[i];

		switch(arg[1]){
			case 's': param->solver = stoi(arg2);
				  break;
			case 'e': param->eta = stof(arg2);
				  break;
			case 'o': param->rho = stof(arg2);
				  break;
			case 'm': param->max_iter = stoi(arg2);
				  break;
			case 'p': param->problem_type = string(arg2);
				  break;
			default:
				  cerr << "unknown option: " << arg << " " << arg2 << endl;
				  exit(0);
		}
	}

	if(i>=args.size())
		exit_with_help();

	param->testFname = argv[i+1];
	i++;
	if( i<args.size() )
		param->modelFname = argv[i+1];
	else{
		param->modelFname = new char[FNAME_LEN];
		strcpy(param->modelFname,"model");
	}
}

inline void construct_factor(Instance* ins, Param* param, vector<UniFactor*>& nodes, vector<BiFactor*>& edges){
	nodes.clear();
	edges.clear();

	//cerr << "constructing uniFactor";
	//construct UniFactors
	int node_count = 0;
	for (int t = 0; t < ins->T; t++){
		UniFactor* node = new UniFactor(ins->node_label_lists[t]->size(), ins->node_score_vecs[t], param);
		nodes.push_back(node);
		//    cerr << ".";
	}
	//cerr << endl;

	//cerr << "constructing biFactor";
	//construct BiFactors
	for (int e = 0; e < ins->edges.size(); e++){
		int e_l = ins->edges[e].first, e_r = ins->edges[e].second;
		ScoreVec* sv = ins->edge_score_vecs[e];
		UniFactor* l = nodes[e_l];
		UniFactor* r = nodes[e_r];
		BiFactor* edge = new BiFactor(l, r, sv, param);
		edges.push_back(edge);
		//if (e % 100 == 0){
		//    cerr << ".";
		//}
	}
	//cerr << "done" << endl;
}

inline Float compute_acc(Instance* ins, vector<UniFactor*> nodes){

	Float hit = 0;
	//for this instance

	//compute hits
	int node_count = 0;
	for (vector<UniFactor*>::iterator it_node = nodes.begin(); it_node != nodes.end(); it_node++, node_count++){	
		UniFactor* node = *it_node;
		//Rounding
		hit+=node->y[ins->labels[node_count]];
	}
	return (Float)hit/ins->T;
}

void print_sol(vector<UniFactor*>& nodes, vector<BiFactor*>& edges, int iter){
    string name = string("GDMM.sol");
    ofstream fout(name+to_string(iter));
    fout << nodes.size() << endl;
    //fout << edges.size() << endl;
    for (vector<UniFactor*>::iterator it = nodes.begin(); it != nodes.end(); it++){
        UniFactor* node = *it;
        fout << node->recent_pred << " ";
    }
    fout << endl;
    fout.close();
}

double struct_predict(MultiLabelProblem* prob, Param* param){
	Float hit = 0.0;
	Float N = 0.0;
	int n = 0;
	stats = new Stats();
	int max_iter = param->max_iter;
	Float score = 0.0;
	for (vector<Instance*>::iterator it_ins = prob->data.begin(); it_ins != prob->data.end(); it_ins++, n++){
		//if (n != param->testno) continue;
		Instance* ins = *it_ins;
		int K = ins->node_label_lists[0]->size();
		MultiUniFactor* node = new MultiUniFactor(K, ins->node_score_vecs[0], param);
		MultiBiFactor* edge = new MultiBiFactor(K, node, ins->edge_score_vecs[0], param);

		/*
		   node->c[0] = 0.286; node->c[1] = 0.366; node->c[2] = -0.128; node->c[3] = 1.0;
		   edge->c[0] = 0; edge->c[1] = -2.1;  edge->c[2] = -0.0293;   edge->c[3] = 0;
		   edge->c[4] = 0; edge->c[5] = 0.0;   edge->c[6] = -0.0222;   edge->c[7] = 0;
		   edge->c[8] = 0; edge->c[9] = 0;     edge->c[10] = 0.0;       edge->c[11] = 0.0;
		   edge->c[12] = 0; edge->c[13] = 0;     edge->c[14] = 0.0;       edge->c[15] = 0.0;
		 */

		/*for (int k1 = 0; k1 < K; k1++){
		  for (int k2 = 0; k2 < K; k2++){
		  edge->c[k1*K+k2] = edge->c[k1*14+k2];
		  }
		  }*/

		/*
		   cerr << "c:\t";
		   for (int i = 0; i < K; i++){
		   cerr << node->c[i] << " ";
		   }
		   cerr << endl;
		   cerr << endl;
		   for (int i = 0; i < K; i++){
		   for (int j = 0; j < K; j++)
		   cerr << setprecision(3) << edge->c[i*K+j] << "\t";
		   cerr << endl;
		   }
		   cerr << endl;
		 */

		Float p_inf = 0.0, acc = 0.0, d_inf = 0.0;
		int iter = 0;
		Float score_t = 0.0;
		int print_period = 1000000;
		Float best_decoded = -1e100;
		while (iter++ < max_iter){
			//cerr << "========================================================" << endl;

			node->search();
			node->subsolve();
			/*while (node->dual_inf() > node->nnz_tol){
			  node->subsolve();
			  node->check_integrity();
			  node->display();
			//edge->display();
			}*/

			stats->uni_act_size += node->act_set.size();
			stats->num_uni++;
			//node->subsolve();

			//node->display();
			//edge->display();
			//node->check_integrity();
			//edge->check_integrity();

			edge->search();
			edge->subsolve();
			/*while (edge->dual_inf() > edge->nnz_tol){
			  edge->subsolve();
			  edge->check_integrity();
			//edge->display();
			}*/

			stats->num_bi++;
			stats->bi_act_size += edge->act_set.size();

			node->update_multipliers();
			edge->update_multipliers();

			prediction_time += get_current_time();
			//node->check_integrity();
			//edge->check_integrity();
			score_t = node->score() + edge->score();

			p_inf = edge->infea();
			d_inf = edge->dual_inf() + node->dual_inf();

			prediction_time -= get_current_time();
			if (debug){
				//cerr << "d_inf=" << d_inf << ", p_inf=" << p_inf << endl;
			}
			if ((iter+1) % print_period == 0){
				if (-score_t > best_decoded){
					best_decoded = -score_t;
				}
				cerr << "iter=" << iter << ", decoded_t=" << (-score_t) << ", best_decoded=" << best_decoded << ", dinf=" << d_inf << ", p_inf=" << p_inf;
				stats->display();
				stats->display_time();
				cerr << ", overall time=" << (prediction_time + (double)get_current_time());
				cerr << endl;
			}
			if (iter >= 5 && d_inf <= param->grad_tol && p_inf <= param->infea_tol){
				break;
			}
		}
		score += score_t;

		//node->display();
		//edge->display();
		//compute hamming loss
		Float T = ins->labels.size();
		for (int k = 0; k < K; k++){
			Float true_yk = (find(ins->labels.begin(), ins->labels.end(), k) != ins->labels.end())? 1.0:0.0;
			Float distance = fabs(true_yk - node->y[k]);
			hit += 1 - distance;
		}
		N += K;
		acc = hit/N;
		//cerr << "hit=" << hit << ", N=" << N << endl; 
		cerr << "@" << n << ": iter=" << iter << ", p_inf=" << p_inf << ", d_inf=" << d_inf << ", acc=" << acc;
		cerr << ", pred={";
		for (vector<pair<Float, int>>::iterator it = node->act_set.begin(); it != node->act_set.end(); it++){
			cerr << "," << it->second << ":" << it->first;
		}
		cerr << "}";
		cerr << ", decoded=" << score;

		//compute score of true answer
		Float true_score = 0;
		cerr << ", true_labels={"; 
		for (int i = 0; i < ins->labels.size(); i++){
			int k = ins->labels[i];
			cerr << "," << k;
			true_score += node->c[k];
		}
		cerr << "}";
		for (int i = 0; i < ins->labels.size(); i++){
			int k1 = ins->labels[i];
			for (int j = i+1; j < ins->labels.size(); j++){
				int k2 = ins->labels[j];
				true_score += edge->c[k1*K+k2];
				true_score += edge->c[k2*K+k1];
			}
		}
		cerr << ", true_score=" << true_score;
		int nnz_c = 0;
		for (int i = 0; i < K*K; i++){
			if (edge->c[i] < 0)
				nnz_c++;
		}
		cerr << ", nnz_c=" << nnz_c;
		stats->display();
		stats->display_time();
		cerr << endl;
	}
	return hit/N; 
}

double struct_predict(Problem* prob, Param* param){
	if (param->problem_type == "multilabel"){
		return struct_predict((MultiLabelProblem*)prob, param);
	}
	if (fabs(param->eta) <= 1e-12){
		//soft consistency constraints
		param->infea_tol = 1e300;
	}
	Float hit = 0.0;
	Float N = 0.0;
	int n = 0;
	stats = new Stats();
	vector<UniFactor*> nodes; 
	vector<BiFactor*> edges;
	Float score = 0.0;
	for (vector<Instance*>::iterator it_ins = prob->data.begin(); it_ins != prob->data.end(); it_ins++, n++){
		Instance* ins = *it_ins;
		stats->construct_time -= get_current_time();
		construct_factor(ins, param, nodes, edges);
		cerr << "#nodes=" << nodes.size() << ", #edges=" << edges.size() << endl;
		int* node_indices = new int[nodes.size()];
		for (int i = 0; i < nodes.size(); i++)
			node_indices[i] = i;
		int* edge_indices = new int[edges.size()];
		for (int i = 0; i < edges.size(); i++)
			edge_indices[i] = i;
		stats->construct_time += get_current_time();
		int iter = 0;
		int max_iter = param->max_iter;
		Float p_inf, d_inf, acc, nnz_msg;
		Float val = 0.0;
		stats->clear();
		int countdown = 0;

		vector<Factor*> factor_seq;
		for (int i = 0; i < ins->T; i++){
			factor_seq.push_back(nodes[i]);
		}
		for (int i = 0; i < edges.size(); i++){
			factor_seq.push_back(edges[i]);
		}

		nnz_msg = 0.0;
		Float score_t = 0.0, rel_score_t = 0.0;
		int print_period = 1;
		Float best_decoded = -1e100;
		while (iter < max_iter){
			score_t = 0.0; rel_score_t = 0.0;
			val = 0.0;
            random_shuffle(node_indices, node_indices+nodes.size());
			for (int n = 0; n < nodes.size(); n++){
				UniFactor* node = nodes[node_indices[n]];

				node->search();

				//cerr << "==============" << endl;
				node->subsolve();
				
				/*while (node->dual_inf() > param->grad_tol){
					node->subsolve();
					node->display();
					int K = node->K;
					//cerr << "K=" << K << endl;
					cerr << "grad:\t";
					for (vector<int>::iterator it = node->act_set.begin(); it != node->act_set.end(); it++)
						cerr << *it << ":" << node->grad[*it] << " ";
					cerr << endl;
					cerr << "dinf=" << node->dual_inf() << endl;
				}*/
				
			}


			random_shuffle(edge_indices, edge_indices+edges.size());
			for (int e = 0; e < edges.size(); e++){
				BiFactor* edge = edges[edge_indices[e]];

				edge->search();

				edge->subsolve();
				/*while (edge->dual_inf() > param->grad_tol){
					edge->subsolve();
				}*/

			}

			/*int factor_count = 0;
			  for (vector<Factor*>::iterator f = factor_seq.begin(); f != factor_seq.end(); f++, factor_count++){
			//if (factor_count % 10000 == 0)    cerr << factor_count << "/" << factor_seq.size() << endl;

			//cerr << "search" << endl;
			(*f)->search();
			//cerr << "subsolve" << endl;
			(*f)->subsolve();
			//cerr << "end" << endl;
			}*/

			for (vector<BiFactor*>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++){
				BiFactor* edge = *it_edge;
				edge->update_multipliers();
			}

			//compute primal inf & dual inf

			//cerr << "==========================" << endl;

			d_inf = 0.0; int dinf_index = -1; int dinf_factor_index = -1;
			int node_count = 0;
			for (vector<UniFactor*>::iterator it_node = nodes.begin(); it_node != nodes.end(); it_node++, node_count++){
				UniFactor* node = *it_node;
				Float gmax_node = node->dual_inf();
				if (gmax_node > d_inf){
					d_inf = gmax_node;
					dinf_index = node->dinf_index;
					dinf_factor_index = node_count;
					//cerr << setprecision(10) << "node" << node_count << ":" << d_inf << ", act_size=" << node->act_set.size() << ", k=" << dinf_index << ", y=" << node->y[dinf_index] << ", grad=" << node->grad[dinf_index] << endl;
				}
				prediction_time += get_current_time();
				score_t += node->score();
                rel_score_t += node->rel_score();
				val += node->func_val();
				stats->uni_act_size += node->act_set.size();
				stats->num_uni++;
				prediction_time -= get_current_time();
			}

			int edge_count = 0;
			p_inf = 0.0;
			for (vector<BiFactor*>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++, edge_count++){
				BiFactor* edge = *it_edge;
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
				score_t += edge->score();
                rel_score_t += edge->rel_score();
				val += edge->func_val();
				stats->bi_act_size += edge->act_set.size();
				stats->ever_nnz_msg_size += (edge->ever_nnz_msg_l.size() + edge->ever_nnz_msg_r.size())/2;
				stats->num_bi++;
				nnz_msg += edge->nnz_msg();
				prediction_time -= get_current_time();
			}
			Float g = 0.0;  
			if (dinf_factor_index != -1){
				if (dinf_factor_index >= nodes.size()){
					edge_count = dinf_factor_index - nodes.size();
					BiFactor* edge = edges[edge_count];
					int k1k2 = dinf_index;
					int k1 = k1k2 / edge->K2, k2 = k1k2 % edge->K2;
					g = -(edge->c[dinf_index] + edge->msgl[k1] + edge->msgr[k2]);
				} else {
					g = nodes[dinf_factor_index]->grad[dinf_index];
				}
			}
			if (p_inf < param->infea_tol && d_inf < param->grad_tol){
				countdown++;
			}
			if ((iter+1) % print_period == 0){
				if (-score_t > best_decoded){
					best_decoded = -score_t;
                    cerr << "best primal obj updated to " << best_decoded << endl;
                    //print_sol(nodes, edges, iter);
				}
				cerr << "iter=" << iter << ", funcval=" << val << ", (-rel_score_t)=" << (-rel_score_t) << ", decoded_t=" << (-score_t) << ", best_decoded=" << best_decoded << ", dinf=" << d_inf << ", p_inf=" << p_inf << ", countdown=" << countdown;
				stats->display();
				stats->display_time();
				cerr << ", overall time=" << (prediction_time + (double)get_current_time());
				cerr << endl;
			}
			iter++;
			if (countdown >= 3){
				break;
			}
		}
		//nnz_msg /= edges.size()*(iter+1);
		score += score_t;

		Float round_score = 0.0;
		int* pred = new int[nodes.size()];
		for (int i = 0; i < nodes.size(); i++){
			UniFactor* node = nodes[i];
			Float r = (Float)rand()/RAND_MAX;
			int K = node->K;
			
			for (int k = 0; k < K; k++){
				if (r <= node->y[k]){
					pred[i] = k;
					break;
				}
				r -= node->y[k];
			}
			round_score -= node->c[pred[i]];
		}
		cerr << "middle" << endl;
		for (int i = 0; i < edges.size(); i++){
			BiFactor* edge = edges[i];
			int l = ins->edges[i].first, r = ins->edges[i].second;
			int K2 = edge->K2;
			round_score -= edge->c[pred[l]*K2 + pred[r]];
		}
	
		prediction_time += get_current_time();
		acc = compute_acc(ins, nodes);
		prediction_time -= get_current_time();

		cerr << "@" << n
			<< ": iter=" << iter
			<< ", T=" << ins->T
			<< ", acc=" << acc
			<< ", score=" << score 
			<< ", round_score=" << round_score
			<< ", infea=" << p_inf
			<< ", dinf=" << d_inf
			<< ", avg_nnz_msg=" << nnz_msg / (iter+1) / edges.size();

		Float ever_nnz_msg = 0;
		for (int i = 0; i < edges.size(); i++)
			ever_nnz_msg += (edges[i]->ever_nnz_msg_l.size() + edges[i]->ever_nnz_msg_r.size());
		ever_nnz_msg /= edges.size();
		cerr << ", ever_nnz_msg=" << ever_nnz_msg;
		stats->display_time();
		cerr << endl; 
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
	Float total_score = 0.0;
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

		total_score +=  max_sum[ins->T-1][pred[ins->T-1]];
		cerr << "@" << n << ": acc=" << (double)(hit-temp_hit)/(ins->T) << " --- " << (Float)hit/N << ", score=" << total_score << endl;

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

	Problem* prob = NULL;
	if (param->problem_type == "chain"){
		prob = new ChainProblem(param);
		prob->construct_data();
		int D = ((ChainProblem*)prob)->D; 
		int K = ((ChainProblem*)prob)->K; 
		cerr << "prob.D=" << D << endl;
		cerr << "prob.K=" << K << endl;
	}
	if (param->problem_type == "network"){
		prob = new CompleteGraphProblem(param);
		prob->construct_data();
		int K = ((CompleteGraphProblem*)prob)->K;
		cerr << "prob.K=" << K << endl;
	}
	if (param->problem_type == "multilabel"){
		prob = new MultiLabelProblem(param);
		prob->construct_data();
		int D = ((MultiLabelProblem*)prob)->D; 
		int K = ((MultiLabelProblem*)prob)->K; 
		cerr << "prob.D=" << D << endl;
		cerr << "prob.K=" << K << endl;
	}
	if (param->problem_type == "uai"){
		prob = new UAIProblem(param);
		prob->construct_data();
	}

    if (param->problem_type == "loguai"){
        prob = new LOGUAIProblem(param);
        prob->construct_data();
    }

	if (prob == NULL){
		cerr << "Need to specific problem type!" << endl;
	}

	cerr << "prob.N=" << prob->data.size() << endl;	
	cerr << "param.rho=" << param->rho << endl;
	cerr << "param.eta=" << param->eta << endl;

	/*
	   double t1 = get_current_time();
	   vector<Float*> cc;
	   for (int i = 0; i < 200; i++){
	   Float* temp_float = new Float[4];
	   cc.push_back(temp_float);
	   }
	   for (int tt = 0; tt < 3000*1000; tt++)
	   for (int i = 0; i < 200; i++){
	   Float* cl = cc[rand()%200];
	   Float* cr = cc[rand()%200];
	   for (int j = 0; j < 4; j++)
	   cl[j] = cr[j];
	   }
	   cerr << get_current_time() - t1 << endl;
	 */

	prediction_time = -get_current_time();
	if (param->solver == 0){
		cerr << "Acc=" << Viterbi_predict((ChainProblem*)prob, param) << endl;
	} 
	if (param->solver == 1){
		cerr << "Acc=" << compute_acc_sparseLP(prob) << endl;
	}
	if (param->solver == 2 || param->solver == 3){
		cerr << "Acc=" << struct_predict(prob, param) << endl;
	}
	prediction_time += get_current_time();
	cerr << "prediction time=" << prediction_time << endl;
	return 0;
}
