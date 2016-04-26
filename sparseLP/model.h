#ifndef MODEL_H
#define MODEL_H

#include "util.h"
#include "extra.h"

class Model{
	
	public:
	Model(char* fname){
		ifstream fin(fname);
		char* line = new char[LINE_LEN];
		//first line, get K
		fin.getline(line, LINE_LEN);
		string line_str(line);
		vector<string> tokens = split(line_str, "=");
		K = stoi(tokens[1]);
		
		//second line, get label_name_list
		fin.getline(line, LINE_LEN);
		line_str = string(line);
		tokens = split(line_str, " ");
		label_name_list = new vector<string>();
		label_index_map = new map<string, Int>();
		//token[0] is 'label', means nothing
		for (Int i = 1; i < tokens.size(); i++){
			label_name_list->push_back(tokens[i]);
			label_index_map->insert(make_pair(tokens[i], (i-1)));
		}
		
		//third line, get D
		fin.getline(line, LINE_LEN);
		line_str = string(line);
		tokens = split(line_str, "=");
		D = stoi(tokens[1]);
		
		//skip fourth line
		fin.getline(line, LINE_LEN);

		//next D lines: read w
		w = new Float*[D];
		for (Int j = 0; j < D; j++){
			w[j] = new Float[K];
			memset(w[j], 0, sizeof(Float)*K);
			fin.getline(line, LINE_LEN);
			line_str = string(line);
			tokens = split(line_str, " ");
			Float* wj = w[j];
			for (vector<string>::iterator it = tokens.begin(); it != tokens.end(); it++){
				vector<string> k_w = split(*it, ":");
				Int k = stoi(k_w[0]);
				wj[k] = stof(k_w[1]);
			}
		}

		//skip next line
		fin.getline(line, LINE_LEN);

		//next K lines: read v
		v = new Float*[K];
		for (Int k1 = 0; k1 < K; k1++){
			v[k1] = new Float[K];
			memset(v[k1], 0, sizeof(Float)*K);
			fin.getline(line, LINE_LEN);
			line_str = string(line);
			tokens = split(line_str, " ");
			Float* v_k1 = v[k1];
			for (vector<string>::iterator it = tokens.begin(); it != tokens.end(); it++){
				vector<string> k2_v = split(*it, ":");
				Int k2 = stoi(k2_v[0]);
				v_k1[k2] = stof(k2_v[1]);
			}
		}
		sparse_v = new vector<pair<Int, Float>>[K];
		for (Int k1 = 0; k1 < K; k1++){
			sparse_v[k1].clear();
			for (Int k2 = 0; k2 < K; k2++){
				if (fabs(v[k1][k2]) > 1e-12){
					sparse_v[k1].push_back(make_pair(k2, v[k1][k2]));
				}
			}
		}
	}

	Model(Float** _w, Float** _v, ChainProblem* prob){
		D = prob->D;
		K = prob->K;
		label_name_list = &(prob->label_name_list);
		label_index_map = &(prob->label_index_map);
		w = _w;
		v = _v;
		sparse_v = new vector<pair<Int, Float>>[K];
		for (Int k1 = 0; k1 < K; k1++){
			sparse_v[k1].clear();
			for (Int k2 = 0; k2 < K; k2++){
				if (fabs(v[k1][k2]) > 1e-12){
					sparse_v[k1].push_back(make_pair(k2, v[k1][k2]));
				}
			}
		}
	}
	
	~Model(){
		for (Int k = 0; k < K; k++){
		       	sparse_v[k].clear();
		}
		delete[] sparse_v;
	}

	Float** w;
	Float** v;
	Int D;
	Int K;
	vector<string>* label_name_list;
	map<string,Int>* label_index_map;
	vector<pair<Int, Float>>* sparse_v;
	
	Float calcAcc_Viterbi(ChainProblem* prob){
		vector<Seq*>* data = &(prob->data);
		Int nSeq = data->size();
		Int N = 0;
		Int hit=0;
		for(Int n=0;n<nSeq;n++){
			
			Seq* seq = data->at(n);
			N += seq->T;
			//compute prediction
			Int* pred = new Int[seq->T];
			Float** max_sum = new Float*[seq->T];
			Int** argmax_sum = new Int*[seq->T];
			for(Int t=0; t<seq->T; t++){
				max_sum[t] = new Float[K];
				argmax_sum[t] = new Int[K];
				for(Int k=0;k<K;k++)
					max_sum[t][k] = -1e300;
			}
			////Viterbi t=0
			SparseVec* xi = seq->features[0];
			for(Int k=0;k<K;k++)
				max_sum[0][k] = 0.0;
			for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
				Float* wj = w[it->first];
				for(Int k=0;k<K;k++)
					max_sum[0][k] += wj[k]*it->second;
			}
			////Viterbi t=1...T-1
			for(Int t=1; t<seq->T; t++){
				//passing message from t-1 to t
				for(Int k1=0;k1<K;k1++){
					Float tmp = max_sum[t-1][k1];
					Float cand_val;
					/*for(Int k2=0;k2<K;k2++){
						 cand_val = tmp + v[k1][k2];
						 if( cand_val > max_sum[t][k2] ){
							max_sum[t][k2] = cand_val;
							argmax_sum[t][k2] = k1;
						 }
					}*/
					for (vector<pair<Int, Float>>::iterator it = sparse_v[k1].begin(); it != sparse_v[k1].end(); it++){
						Int k2 = it->first; 
						cand_val = tmp + it->second;
						if( cand_val > max_sum[t][k2] ){
							max_sum[t][k2] = cand_val;
							argmax_sum[t][k2] = k1;
						}	
					}
				}
				//adding unigram factor
				SparseVec* xi = seq->features[t];
				for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++)
					for(Int k2=0;k2<K;k2++)
						max_sum[t][k2] += w[it->first][k2] * it->second;
			}
			////Viterbi traceback
			pred[seq->T-1] = argmax( max_sum[seq->T-1], K );
			for(Int t=seq->T-1; t>=1; t--){
				pred[t-1] = argmax_sum[t][ pred[t] ];
			}

			//for (int t = 0; t < seq->T; t++)
			//	cerr << "max_score @"<< t <<"= " << max_sum[t][pred[t]] << endl;
			
			/*cerr << "c[27]=" << max_sum[0][27] << endl;
			cerr << "v[27][9]=" << v[27][9] << endl;
			double c9 = 0.0;
			SparseVec* xii = seq->features[1];
			for(SparseVec::iterator it=xii->begin(); it!=xii->end(); it++)
					c9 += w[it->first][9] * it->second;
			cerr << "c[9]=" << c9 << endl;*/
			///////////
			/*for (Int t = 0; t < seq->T; t++){
				cerr << pred[t] << " ";
			}
			cerr << endl;*/
			///////////			

			//compute accuracy
			int temp_hit = hit;
			for(Int t=0;t<seq->T;t++){
				if( label_name_list->at(pred[t]) == prob->label_name_list[seq->labels[t]] )
				//if( pred[t] == seq->labels[t] )
					hit++;
			}
		
			//cerr << (double)(hit-temp_hit)/(seq->T) << endl;

			for(Int t=0; t<seq->T; t++){
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
};

#endif
