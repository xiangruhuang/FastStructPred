#include "chain.h"
#include "BDMMsolve.h"
#include "BCFWsolve.h"

double overall_time = 0.0;

void exit_with_help(){
	cerr << "Usage: ./train (options) [train_data] (model)" << endl;
	cerr << "options:" << endl;
	cerr << "-s solver: (default 0)" << endl;
	cerr << "	0 -- BDMM" << endl;
	//cerr << "	1 -- Active Block Coordinate Descent" << endl;
	cerr << "	1 -- BCFW" << endl;
	//cerr << "-l lambda: L1 regularization weight (default 1.0)" << endl;
	cerr << "-c cost: cost of each sample (default 1)" << endl;
	//cerr << "-r speed_up_rate: using 1/r fraction of samples (default min(max(DK/(log(K)nnz(X)),1),d/5) )" << endl;
	cerr << "-q split_up_rate: choose 1/q fraction of [K]" << endl;
	cerr << "-m max_iter: maximum number of iterations allowed (default 20)" << endl;
	//cerr << "-i im_sampling: Importance sampling instead of uniform (default not)" << endl;
	cerr << "-o heldout_period: period(#iters) to report heldout accuracy (default 10)" << endl;
	//cerr << "-g max_select: maximum number of greedy-selected dual variables per sample (default 1)" << endl;
	//cerr << "-p post_train_iter: #iter of post-training w/o L1R (default 0)" << endl;
	cerr << "-h heldout data set: use specified heldout data set" << endl;
	cerr << "-b brute_force search: use naive search (default off)" << endl;
	cerr << "-w write_model_period: write model file every (arg) iterations (default max_iter)" << endl;
	cerr << "-e early_terminate: stop if heldout accuracy doesn't increase in (arg) iterations (need -h) (default 3)" << endl;
	cerr << "-a admm_step_size: admm update step size (default 1.0) " << endl;
	cerr << "-u non-fully corrective update: use non-fully corrective update (default off) " << endl;
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
//			case 'l': param->lambda = atof(argv[i]);
//				  break;
			case 'c': param->C = atof(argv[i]);
				  break;
			case 'm': param->max_iter = atoi(argv[i]);
				  break;
//			case 'g': param->max_select = atoi(argv[i]);
//				  break;
//			case 'r': param->speed_up_rate = atoi(argv[i]);
//				  break;
//			case 'i': param->using_importance_sampling = true; --i;
//				  break;
			case 'q': param->split_up_rate = atoi(argv[i]);
				  break;
//			case 'p': param->post_solve_iter = atoi(argv[i]);
//				  break;
			case 'h': param->heldoutFname = argv[i];
				  break;
			case 'b': param->using_brute_force = true; --i;
				  break;
			case 'w': param->write_model_period = atoi(argv[i]);
				  break;
			case 'e': param->early_terminate = atoi(argv[i]);
				  break;
			case 'a': param->admm_step_size = atof(argv[i]);
				  break;
			case 't': param->eta = atof(argv[i]);
				  break;
			case 'u': param->do_subSolve = false; --i;
				  break;
			case 'o': param->heldout_period = atoi(argv[i]);
				  break;
			default:
				  cerr << "unknown option: -" << argv[i-1][1] << endl;
				  exit(0);
		}
	}

	if(i>=argc)
		exit_with_help();

	param->trainFname = argv[i];
	i++;
	if (param->write_model_period == 0){
		param->write_model_period = param->max_iter;
	}
	if( i<argc )
		param->modelFname = argv[i];
	else{
		param->modelFname = new char[FNAME_LEN];
		strcpy(param->modelFname,"model");
	}
}

int main(int argc, char** argv){
	
	if( argc < 1+1 ){
		exit_with_help();
	}

	Param* param = new Param();
	parse_cmd_line(argc, argv, param);
	param->prob = new ChainProblem(param->trainFname);
	if (param->heldoutFname != NULL){
		cerr << "using heldout data set: " << param->heldoutFname << ", early_terminate=" << param->early_terminate << endl;
		param->heldout_prob = new ChainProblem(param->heldoutFname);
	}
	param->prob->label_random_remap();
	if (param->heldoutFname != NULL)
		param->heldout_prob->label_random_remap();
	overall_time = -get_current_time();

	cerr << "D=" << param->prob->D << endl;
	cerr << "K=" << param->prob->K << endl;
	cerr << "N=" << param->prob->N << endl;
	cerr << "nSeq=" << param->prob->data.size() << endl;

	if (param->solver == 0){
		BDMMsolve* solver = new BDMMsolve(param);
		Model* model = solver->solve();
		//cerr << "Acc=" << model->calcAcc_Viterbi(param->prob) << endl;
	} else {
		BCFWsolve* solver = new BCFWsolve(param);
		Model* model = solver->solve();
		cerr << "Acc=" << model->calcAcc_Viterbi(param->prob) << endl;
		model->writeModel(param->modelFname);
	}
	
	overall_time += get_current_time();
	cerr << "overall time=" << overall_time << endl;
	
	return 0;
}
