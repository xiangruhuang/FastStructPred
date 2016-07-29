#include "problem.h"
#include <time.h>

double prediction_time = 0.0;
bool debug = false;

void exit_with_help(){
	cerr << "Usage: ./predict (options) [testfile] [model]" << endl;
	cerr << "options:" << endl;
	cerr << "-p problem_type: " << endl;
	cerr << "   chain -- sequence labeling problem" << endl;
	cerr << "   network -- network matching problem" << endl;
	cerr << "   uai -- uai format problem" << endl;
	cerr << "   multilabel -- multilabel problem" << endl;
    cerr << "-o output UAI file path" << endl;
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
			case 'e': param->eta = stof(arg2);
				  break;
			case 'p': param->problem_type = arg2;
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

void convert2uai(MultiLabelProblem* prob, string fname){
    vector<Instance*>& data = prob->data;
    int K = prob->K;
    int T = K;
    ofstream fout(fname);
    fout << "MARKOV" << endl;
    Instance* ins = data[0];
    fout << ins->T << endl; 
    for (int t = 0; t < T; t++){
        if (t != 0)
            fout << " ";
        fout << 2;
    }
    fout << endl;
    fout << T+K*(K-1)/2 << endl;
    for (int t = 0; t < T; t++){
        fout << "1 " << t << endl;
    }
    for (int k1 = 0; k1 < K; k1++){
        for (int k2 = k1+1; k2 < K; k2++){
            fout << "2 " << k1 << " " << k2 << endl;
        }
    }
    for (int t = 0; t < T; t++){
        fout << 2 << endl;
        Float v_t = -ins->node_score_vecs[0][t];
        fout << "1.0 " << exp(v_t) << endl;
        fout << endl;
    }
    for (int k1 = 0; k1 < K; k1++){
        for (int k2 = k1+1; k2 < K; k2++){
            fout << 4 << endl;
            fout << "1.0 1.0" << endl;
            Float v_k1_k2 = -ins->edge_score_vecs[0]->c[k1*K+k2];
            fout << "1.0 " << exp(v_k1_k2) << endl;
            fout << endl;
        }
    }
    fout.close();
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
	    convert2uai((MultiLabelProblem*)prob, param->convertFname);
    }
	return 0;
}
