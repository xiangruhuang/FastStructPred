#include "problem.h"
#include "LPsparse.h"
#include "model.h"
#include "factor.h"

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

inline void construct_factor(Instance* ins, Model* model, Param* param, vector<uni_factor*>& nodes, vector<bi_factor*>& edges){
	//construct uni_factors
	nodes.clear();
	edges.clear();
	//int node_count = 0;
	//int n = prob->data.size();
	//for (vector<Instance*>::iterator it_ins = prob->data.begin(); it_ins != prob->data.end(); it_ins++){
	//	Instance* ins = *it_ins;
		//for this instance
		
	//offset of this subgraph
	//int offset = node_count;

	//construct uni_factors
	for (vector<SparseVec*>::iterator it_node = ins->features.begin(); it_node != ins->features.end(); it_node++){
		uni_factor* node = new uni_factor(model->K, *it_node, model->w, param);
		nodes.push_back(node);
	}

	//construct bi_factors
	for (vector<pair<int, int>>::iterator it_e = ins->edges.begin(); it_e != ins->edges.end(); it_e++){
		uni_factor* l = nodes[it_e->first];
		uni_factor* r = nodes[it_e->second];
		bi_factor* edge = new bi_factor(model->K, l, r, model->v, param);
		edges.push_back(edge);
	}
	//}
	//assert(n + edges.size() == nodes.size());
}

inline Float compute_acc(Problem* prob, Instance* ins, Model* model, vector<uni_factor*> nodes){

	int node_count = 0;
	int hit = 0;
	int N = 0;
	//for this instance

	//compute hits
	for (vector<SparseVec*>::iterator it_node = ins->features.begin(); it_node != ins->features.end(); it_node++, node_count++){	
		uni_factor* node = nodes[node_count];
		//Rounding
		int label = -1;
		double s = 0.0;
		for (int k = 0; k < model->K; k++){
			s += node->y[k];
		}
		Float r = s*((Float)rand()/RAND_MAX);
		for (int k = 0; k < model->K; k++){
			if (r <= node->y[k]){
				label = k;
				break;
			} else {
				r -= node->y[k];
			}
		}
		assert(label != -1 /* prediction should be valid */);
		if (prob->label_name_list[ins->labels[node_count]] == model->label_name_list->at(label)){
			hit++;
		}
	}
	return (Float)hit/ins->T;
}

void struct_predict(Problem* prob, Model* model, Param* param){
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
		construct_factor(ins, model, param, nodes, edges);
		construct_time += omp_get_wtime();
		int iter = 0;
		int max_iter = param->max_iter;
		Float score = 0.0;
		Float p_inf = 0.0;
		Float val = 0.0;
		while (iter < max_iter){
			score = 0.0;
			p_inf = 0.0;
			val = 0.0;
			for (vector<uni_factor*>::iterator it_node = nodes.begin(); it_node != nodes.end(); it_node++){
				uni_factor* node = *it_node;

				uni_search_time -= omp_get_wtime();
				node->search();
				uni_search_time += omp_get_wtime();

				uni_subsolve_time -= omp_get_wtime();
				node->subsolve();
				uni_subsolve_time += omp_get_wtime();

				score += node->score();
				val += node->func_val();
				//node->display();
			}

			for (vector<bi_factor*>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++){
				bi_factor* edge = *it_edge;

				bi_search_time -= omp_get_wtime();
				edge->search();
				bi_search_time += omp_get_wtime();

				bi_subsolve_time -= omp_get_wtime();
				edge->subsolve();
				bi_subsolve_time += omp_get_wtime();

				score += edge->score();
				val += edge->func_val();
				p_inf += edge->infea();
				//edge->display();
			}

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
		//cerr << endl;
		Float acc = compute_acc(prob, ins, model, nodes);
		/*cerr << "iter=" << iter 
			<< ", acc=" << acc
			<< ", score=" << score 
			<< ", val=" << val
			<< ", infea=" << p_inf
			<< endl;*/
		hit += acc*ins->T;
		N += ins->T;
		//cerr << "_Acc=" << hit/N << endl;
	}
	cerr << "uni_search=" << uni_search_time
		<< ", uni_subsolve=" << uni_subsolve_time
		<< ", bi_search=" << bi_search_time
		<< ", bi_subsolve=" << bi_subsolve_time 
		<< ", maintain=" << maintain_time 
		<< ", construct=" << construct_time 
		<< ", Acc=" << hit/N << endl;
}

int main(int argc, char** argv){
	if (argc < 2){
		exit_with_help();
	}

/*	Float* y = new Float[3]; 
	Float* b = new Float[3]; b[0] = 0; b[1] = 0; b[2] = 0;
	solve_simplex(3, y, b);
	for (int i = 0; i < 3; i++)
		cerr << y[i] << endl;
	exit(0);
*/	
	Param* param = new Param();
	parse_cmd_line(argc, argv, param);
	Model* model = new Model(param->modelFname);
	cerr << "D=" << model->D << endl;
	cerr << "K=" << model->K << endl;
	
	Problem* prob = new Problem(param->testFname, "chain"); 
	//prob->label_random_remap();
	cerr << "prob.D=" << prob->D << endl;
	cerr << "prob.K=" << prob->K << endl;
	cerr << "prob.N=" << prob->data.size() << endl;
	cerr << "param.rho=" << param->rho << endl;
	cerr << "param.eta=" << param->eta << endl;

	prediction_time -= omp_get_wtime();
	if (param->solver == 0){
		ChainProblem* chain = new ChainProblem(param->testFname);
		cerr << "Acc=" << model->calcAcc_Viterbi(chain) << endl;
	} else {
		//cerr << "Acc=" << compute_acc_sparseLP(model, prob) << endl;
		struct_predict(prob, model, param);
	}
	prediction_time += omp_get_wtime();
	cerr << "prediction time=" << prediction_time << endl;
	return 0;	
}
