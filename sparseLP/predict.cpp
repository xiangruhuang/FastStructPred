#include "problem.h"
#include "LPsparse.h"
#include "model.h"

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
		cerr << "Acc=" << compute_acc_sparseLP(model, prob) << endl;
	}
	return 0;	
}
