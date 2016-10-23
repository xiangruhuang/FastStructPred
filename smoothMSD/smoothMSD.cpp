#include "problem.h"
#include "util.h"
#include <iostream>

using namespace std;

double prediction_time = 0.0;

void exit_with_help(){
	cerr << "./smoothMSD (options) [test_file]" << endl;
	cerr << "-g <gamma>: set gamma, default(1.0)" << endl;
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
		if( arg[0] != '-' )
			break;

		if( ++i >= args.size() )
			exit_with_help();

		string arg2 = args[i];

		switch(arg[1]){
			case 'g': param->gamma = stof(arg2);
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

	if(i>=args.size()){
		exit_with_help();
	}

	param->testFname = argv[i+1];
}

//get e^b proportional to e^a, and normalized s.t. \|e^b\|_1 = 1
void normalize(Float* a, Float* b, int size){
	Float max = -1e100;
	for (int i = 0; i < size; i++){
		if (a[i] > max){
			max = a[i];
		}
	}
	Float shift = max-20;
	Float sum = 0.0;
	for (int i = 0; i < size; i++){
		b[i] = a[i] - shift;
		if (b[i] < -100){
			b[i] = -100;
			continue;
		}
		sum += exp(b[i]);
	}
	
	for (int i = 0; i < size; i++){
		b[i] -= log(sum);
	}
}

// compute log \sum_i e^{x[i]}
Float score(Float* x, int size){
	//cerr << "x:";
	Float max = -1e100;
	for (int i = 0; i < size; i++){
		if (x[i] > max){
			max = x[i];
		}
	//	cerr << x[i] << " ";
	}
	//cerr << endl;
	Float shift = max-20;
	Float sum = 0.0;
	for (int i = 0; i < size; i++){
		Float e = x[i] - shift;
		if (e < -100){
			continue;
		}
		sum += exp(e);
	}
	//cerr << "sum =" << sum << ", score = " << shift+log(sum) << endl;
	return shift+log(sum);
}

void smoothMSD(Problem* prob){
	
	Instance* ins = prob->data[0];
	Float gamma = prob->param->gamma;
	int max_iter = prob->param->max_iter;
	int iter = 0;	
	int E = ins->edges.size();
	int N = ins->node_score_vecs.size();
	vector<Float*> pow, logx;
	for (int i = 0; i < N; i++){
		int Ki = ins->node_label_lists[i]->size();
		Float* pi = new Float[Ki];
		Float* ci = ins->node_score_vecs[i];
		for (int k = 0; k < Ki; k++){
			pi[k] = -ci[k]/gamma;
		}
		
		pow.push_back(pi);
		Float* logxi = new Float[Ki];
		logx.push_back(logxi);
	}
	vector<Float*> msgl, msgr;
	for (int e = 0; e < E; e++){
		int l = ins->edges[e].first, r = ins->edges[e].second;
		int Kl = ins->node_label_lists[l]->size();
		int Kr = ins->node_label_lists[r]->size();
		Float* msg = new Float[Kl];
		memset(msg, 0.0, sizeof(Float)*Kl);
		msgl.push_back(msg);
		msg = new Float[Kr];
		memset(msg, 0.0, sizeof(Float)*Kr);
		msgr.push_back(msg);
	}

	Float best_primal_obj = -1e100;
	int* pred = new int[N];

	Float last_dual = 0.0;
	int counter = 0;
	vector<int> edges;
	for (int e = 0; e < E; e++){
		edges.push_back(e);
	}
	while (iter < max_iter){
		Float int_gap = 0.0, infea_inf = -1e100;
		Float pow_l1 = 0.0;
		//compute logx[i], forall i
		for (int i = 0; i < N; i++){
			int Ki = ins->node_label_lists[i]->size();
			normalize(pow[i], logx[i], Ki);
			//Float sum = 0;
			//for (int k = 0; k < Ki; k++){
			//	sum += exp(logx[i][k]);
			//}
			//assert(fabs(sum - 1.0) < 1e-6);
			
			//cerr << i << "(x):";
			//for (int k = 0; k < Ki; k++){
			//	cerr << "\t" << exp(logx[i][k]);
			//}
			//cerr << endl;

			//cerr << i << "(pow):";
			//for (int k = 0; k < Ki; k++){
			//	cerr << "\t" << gamma*pow[i][k];
			//}
			//cerr << endl;
			
		}
		Float dual_obj = 0.0, relaxed_dual = 0.0;
		Float primal_obj = 0.0, relaxed_primal = 0.0;
		//compute unigram objectives: 
		for (int i = 0; i < N; i++){
			int Ki = ins->node_label_lists[i]->size();
			Float* logxi = logx[i];
			Float* ci = ins->node_score_vecs[i];
			int max_index = 0;
			for (int k = 0; k < Ki; k++){
				if (logxi[k] > logxi[max_index]){
					max_index = k;
				}
				pow_l1 += fabs(pow[i][k]);
			}
			pred[i] = max_index;
			dual_obj += gamma*pow[i][pred[i]];
			//cerr << "dual_obj += " << gamma*pow[i][pred[i]] << endl;
			primal_obj += -ci[pred[i]];
			int_gap += fabs(1-exp(logxi[pred[i]]));
			
			Float score_i = gamma*score(pow[i], Ki);
			relaxed_dual += score_i;
			for (int k = 0; k < Ki; k++){
				if (logxi[k] < -100){
					continue;
				}
				Float xi_k = exp(logxi[k]);
				relaxed_primal += xi_k*(-ci[k]);
				relaxed_primal += -gamma*xi_k*logxi[k];
			}
		}
		random_shuffle(edges.begin(), edges.end());		
		Float grad_l2 = 0.0;
		//iterate through all edges, compute bigram objectives
		Float lowest_powe = 1e200;
		for (int ei = 0; ei < E; ei++){
			int e = edges[ei];
			int l = ins->edges[e].first, r = ins->edges[e].second;
			int Kl = ins->node_label_lists[l]->size();
			int Kr = ins->node_label_lists[r]->size();	
			Float* ce = ins->edge_score_vecs[e]->c;
			Float* logy = new Float[Kl*Kr];
			Float* pow_e = new Float[Kl*Kr];
			Float* msgl_e = msgl[e];
			Float* msgr_e = msgr[e];
			Float* logxl = logx[l]; 
			Float* logxr = logx[r]; 
			for (int kk = 0; kk < Kl*Kr; kk++){
				pow_e[kk] = (-ce[kk] - msgl_e[kk/Kr] - msgr_e[kk % Kr])/gamma;
				pow_l1 += fabs(pow_e[kk]);
				if (pow_e[kk] < lowest_powe){
					lowest_powe = pow_e[kk];
				}
			}
			Float score_i = gamma*score(pow_e, Kl*Kr);
			relaxed_dual += score_i;
			//compute log(y_e)
			normalize(pow_e, logy, Kl*Kr);
			
			int pred_e = pred[l]*Kr + pred[r];
			primal_obj += -ce[pred_e];
			int_gap += fabs(1-exp(logy[pred_e]));
			for (int kk = 0; kk < Kl*Kr; kk++){
				int kl = kk/Kr, kr = kk%Kr;
				if (logy[kk] < -100){
					continue;
				}
				Float y_kk = exp(logy[kk]);
				relaxed_primal += -ce[kk]*y_kk;
				relaxed_primal += -gamma*y_kk*logy[kk];
			}
			int max_yindex = 0;
			for (int kk = 0; kk < Kl*Kr; kk++){
				if (logy[kk] > logy[max_yindex]){
					max_yindex = kk;
				}
			}
			dual_obj += gamma*pow_e[max_yindex];
			//cerr << "dual += " << gamma*pow_e[max_yindex] << endl;
			
			//update block: message(il, e)
			Float* powl = pow[l];
			for (int k1 = 0; k1 < Kl; k1++){
				Float y_k1 = 0.0;
				Float shift = -1e200;
				for (int k2 = 0; k2 < Kr; k2++){
					if (logy[k1*Kr + k2] > shift){
						shift = logy[k1*Kr + k2];
					}
				}
				shift -= 20;
				for (int k2 = 0; k2 < Kr; k2++){
					if (logy[k1*Kr+k2]-shift < -100){
						continue;
					}
					y_k1 += exp(logy[k1*Kr+k2]-shift);
				}
				Float delta_msg = gamma/2*(log(y_k1) + shift - logxl[k1]);
				if (delta_msg/(gamma/2.0) > infea_inf){
					infea_inf = delta_msg/(gamma/2.0);
				}
				grad_l2 += delta_msg*delta_msg;
				msgl_e[k1] += delta_msg;
				powl[k1] += delta_msg/gamma;
			}

			//update block: message(ir, e)
			Float* powr = pow[r];
			for (int k2 = 0; k2 < Kr; k2++){
				Float y_k2 = 0.0;
				Float shift = -1e200;
				for (int k1 = 0; k1 < Kl; k1++){
					if (logy[k1*Kr + k2] > shift){
						shift = logy[k1*Kr + k2];
					}
				}
				shift -= 20;
				for (int k1 = 0; k1 < Kl; k1++){
					if (logy[k1*Kr+k2]-shift < -100){
						continue;
					}
					y_k2 += exp(logy[k1*Kr+k2]-shift);
				}
				Float delta_msg = gamma/2*(log(y_k2) + shift - logxr[k2]);
				if (fabs(delta_msg/(gamma/2.0)) > infea_inf){
					infea_inf = fabs(delta_msg/(gamma/2.0));
				}
				grad_l2 += delta_msg*delta_msg;
				msgr_e[k2] += delta_msg;
				powr[k2] += delta_msg/gamma;
			}
			delete logy;
			delete pow_e;
		}
		grad_l2 = sqrt(grad_l2);
		if (best_primal_obj < primal_obj){
			best_primal_obj = primal_obj;
		}
		cerr << "iter=" << iter << ", primal_obj=" << primal_obj << ", best_primal_obj=" << best_primal_obj << ", dual_obj=" << dual_obj << ", relaxed_primal=" << relaxed_primal << ", relaxed_dual=" << relaxed_dual << ", grad_l2=" << grad_l2 << ", infea_inf(ratio=max/min)=e^(" << infea_inf << "), int_gap=" << int_gap << ", pow_l1=" << pow_l1 << ", time=" << get_current_time() + prediction_time << endl;
		if (infea_inf < 1e-6 && fabs(relaxed_dual - relaxed_primal) < 1e-6){
			break;
		}
		if (dual_obj - best_primal_obj < 1e-6){
			break;
		}
		/*if (fabs(last_dual - relaxed_dual) < 1e-3){
			counter++;
		} else {
			counter = 0;
		}
		if (counter > 10){
			break;
		}*/
		last_dual = relaxed_dual;
		
		iter++;
				
	}

	delete pred;
	for (int e = 0; e < E; e++){
		delete msgl[e];
		delete msgr[e];
	}
	for (int i = 0; i < N; i++){
		delete pow[i];
		delete logx[i];
	}
}

int main(int argc, char** argv){
	if (argc < 2){
		exit_with_help();
	}
	
	prediction_time = -get_current_time();
	srand(time(NULL));
	Param* param = new Param();
	parse_cmd_line(argc, argv, param);
	
	Problem* prob = NULL;
	if (param->problem_type == "uai"){
		prediction_time += get_current_time();
		prob = new UAIProblem(param);
		prediction_time -= get_current_time();
		prob->construct_data();
	}
	if (param->problem_type == "loguai"){
		prediction_time += get_current_time();
		prob = new LOGUAIProblem(param);
		prediction_time -= get_current_time();
		prob->construct_data();
	}
	cerr << "prob.N=" << prob->data.size() << endl;	
	cerr << "param.gamma=" << param->gamma << endl;
	
	smoothMSD(prob);
	
}
