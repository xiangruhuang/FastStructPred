#include "problem.h"
//#include "LPsparse.h"
#include "model.h"
#include "factor.h"

double prediction_time = 0.0;

void exit_with_help(){
	cerr << "Usage: ./train (options) [testfile] [model]" << endl;
	cerr << "options:" << endl;
	cerr << "-s solver: (default 0)" << endl;
	cerr << "	0 -- Viterbi(chain)" << endl;
	cerr << "	1 -- sparseLP" << endl;
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

void construct_factor(Problem* prob, Model* model, Param* param, vector<uni_factor*>& nodes, vector<bi_factor*>& edges){
	//construct uni_factors
	nodes.clear();
	edges.clear();
	int node_count = 0;
	for (vector<Instance*>::iterator it_ins = prob->data.begin(); it_ins != prob->data.end(); it_ins++){
		Instance* ins = *it_ins;
		//for this instance
		
		//offset of this subgraph
		int offset = node_count;
		
		//construct uni_factors
		for (vector<SparseVec*>::iterator it_node = ins->features.begin(); it_node != ins->features.end(); it_node++){
			uni_factor* node = new uni_factor(model->K, *it_node, model->w, param);
			nodes.push_back(node);
		}
		
		//construct bi_factors
		for (vector<pair<int, int>>::iterator it_e = ins->edges.begin(); it_e != ins->edges.end(); it_e++){
			uni_factor* l = nodes[offset + it_e->first];
			uni_factor* r = nodes[offset + it_e->second];
			bi_factor* edge = new bi_factor(model->K, l, r, model->v, param);
			edges.push_back(edge);
		}
	}
	
}

Float compute_acc(Problem* prob, Model* model, vector<uni_factor*> nodes){

	int node_count = 0;
	int hit = 0;
	int N = 0;
	for (vector<Instance*>::iterator it_ins = prob->data.begin(); it_ins != prob->data.end(); it_ins++){
		Instance* ins = *it_ins;
		//for this instance
		
		//offset of this subgraph
		int offset = node_count;
		
		//compute hits
		for (vector<SparseVec*>::iterator it_node = ins->features.begin(); it_node != ins->features.end(); it_node++, node_count++){
			
			uni_factor* node = nodes[node_count];
			//Rounding
			Float r = (Float)rand()/RAND_MAX;
			int label = -1;
			for (int k = 0; k < model->K; k++){
				if (r <= node->y[k]){
					label = k;
					break;
				} else {
					r -= node->y[k];
				}
			}
			assert(label != -1 /* prediction should be valid */);
			if (prob->label_name_list[ins->labels[node_count - offset]] == model->label_name_list->at(label)){
				hit++;
			}
			N++;
		}
	}
	return (Float)hit/N;
}

void struct_predict(Problem* prob, Model* model, Param* param){
	vector<uni_factor*> nodes; 
	vector<bi_factor*> edges;
	construct_factor(prob, model, param, nodes, edges);
	
	int iter = 0;
	int max_iter = param->max_iter;
	while (iter < max_iter){
		for (vector<uni_factor*>::iterator it_node = nodes.begin(); it_node != nodes.end(); it_node++){
			uni_factor* node = *it_node;
			node->search();
			node->subsolve();
		}
		
		for (vector<bi_factor*>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++){
			bi_factor* edge = *it_edge;
			edge->search();
			edge->subsolve();
		}
		
		for (vector<bi_factor*>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++){
			bi_factor* edge = *it_edge;
			edge->update_multipliers();
		}
		iter++;
		cerr << "iter=" << iter << ", acc=" << compute_acc(prob, model, nodes) << endl;
	}

}

int main(int argc, char** argv){
	if (argc < 2){
		exit_with_help();
	}
	
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

	if (param->solver == 0){
		ChainProblem* chain = new ChainProblem(param->testFname);
		cerr << "Acc=" << model->calcAcc_Viterbi(chain) << endl;
	} else {
		struct_predict(prob, model, param);
	}
	return 0;	
}
