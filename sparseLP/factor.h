//prediction

#include "util.h"

//unigram factor, y follows simplex constraints
class uni_factor{
	public:
	//fixed
	int K;
	Float rho;
	Float* c; // score vector, c[k] = -<w_k, x>
	
	//maintained
	Float* grad;
	Float* y;
	bool* inside;
	vector<Float*> msgs;
	vector<int> act_set;
    vector<int> ever_act_set;
    bool* is_ever_act;

	inline uni_factor(int _K, Float* _c, Param* param){
		K = _K;
		rho = param->rho;

		//compute score vector
		c = _c;

		//cache of gradient
		grad = new Float[K];
		memset(grad, 0.0, sizeof(Float)*K);

		//relaxed prediction vector
		y = new Float[K];
		memset(y, 0.0, sizeof(Float)*K);

		inside = new bool[K];
		memset(inside, false, sizeof(bool)*K);
        act_set.clear();
        ever_act_set.clear();
        is_ever_act = new bool[K];
        memset(is_ever_act, false, sizeof(bool)*K);
		msgs.clear();

		//temporary
		//fill_act_set();
	}

	~uni_factor(){
		delete[] y;
		delete[] grad;
		delete[] inside;
        delete[] is_ever_act;
		act_set.clear();
		msgs.clear();
	}
	
	void fill_act_set(){
		act_set.clear();
        ever_act_set.clear();
		for (int k = 0; k < K; k++){
			act_set.push_back(k);
            ever_act_set.push_back(k);
            is_ever_act[k] = true;
			inside[k] = true;
		}
	}

	inline void search(){
		//compute gradient of y_i
		for (int k = 0; k < K; k++){
			grad[k] = c[k];
		}
		for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
			Float* msg = *m;
			for (int k = 0; k < K; k++)
				grad[k] -= rho * msg[k];
		}
		Float gmax = 0.0;
		int max_index = -1;
		for (int k = 0; k < K; k++){
			if (inside[k]) continue;
			//if not inside, y_k is guaranteed to be zero, and y_k >= 0 by constraint
			if (grad[k] > 0) continue;
			if (-grad[k] > gmax){
				gmax = -grad[k];
				max_index = k;
			}
		}
		
        if (max_index != -1){
			act_set.push_back(max_index);
            if (!is_ever_act[max_index]){
                is_ever_act[max_index] = true;
                ever_act_set.push_back(max_index);
            }
			inside[max_index] = true;
		}
	}


	//	min_{y \in simplex} <c, y> + \rho/2 \sum_{msg \in msgs} \| (msg + y) - y \|_2^2
	// <===>min_{y \in simplex} \| y - 1/|msgs| ( \sum_{msg \in msgs} (msg + y) - 1/\rho c ) \|_2^2
	// <===>min_{y \in simplex} \| y - b \|_2^2	
	inline void subsolve(){
		if (act_set.size() == 0)
			return;
		Float* y_new = new Float[act_set.size()];
		//cerr << "before subsolve, val=" << func_val() << endl;
		int act_count = 0;
		if (msgs.size() == 0 || fabs(rho) < 1e-12){
			//min_y <c, y>
			Float cmin = 1e300;
			int min_index = -1;
			act_count = 0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k = *it;
				if (c[k] < cmin){
					cmin = c[k];
					min_index = act_count;
				}
			}
			assert(min_index != -1 && "smallest coordinate should exist");
			
			memset(y_new, 0.0, sizeof(Float)*act_set.size());
			y_new[min_index] = 1.0;
		} else {
			Float* b = new Float[act_set.size()];
			memset(b, 0.0, sizeof(Float)*act_set.size());
			act_count = 0;
			for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
				Float* msg = *m;
				act_count = 0;
				for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
					int k = *it;
					b[act_count] += msg[k] + y[k];
				}
			}
			int n = msgs.size();
			act_count = 0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k = *it;
				b[act_count] -= c[k]/rho;
				b[act_count] /= (Float)n;
			}

			solve_simplex(act_set.size(), y_new, b);
			/*act_count = 0;
			  for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
			  int k = *it;
			  cerr << k << ":" << b[act_count] << " ";
			// << "=" << (y[k1][k2]) << "-" << (rho*(msgl[k1])) << "-" << (rho*(msgr[k2])) << "+" << (rho*(sumrow[k1])) << "+" << (rho*(sumcol[k2])) << " ";
			}
			cerr << endl;
			 *///////////////////////
			delete[] b;
		}
		
		for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
			Float* msg = *m;
			act_count = 0;
			if (fabs(y_new[act_count]) < 1e-12){
				y_new[act_count] = 0.0;
			}
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k = *it;
				Float delta_y = y_new[act_count] - y[k];
				msg[k] -= delta_y; // since msg = M Y - y + \mu
			}
		}

		vector<int> next_act_set;
		next_act_set.clear();
		act_count = 0;
		for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
			int k = *it;
			y[k] = y_new[act_count];
			//shrink
			if (y[k] > 1e-12){
				next_act_set.push_back(k);
			} else {
				inside[k] = false;
			}
		}
		act_set = next_act_set;
		
		//cerr << "after subsolve, val=" << func_val() << endl;
		
		delete[] y_new;
	}

	//goal: minimize score
	Float score(){
		Float score = 0.0;
		for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
            int k = *it;
			score += c[k]*y[k];
		}
		return score;
	}

	// F = c^T y + \rho/2 \sum_{msg} \|msg \|^2
	Float func_val(){
		Float val = 0.0;
		for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
            int k = *it;
			val += c[k]*y[k];
		}
		for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
			Float* msg = *m;
			for (int k = 0; k < K; k++){
				val += rho/2 * msg[k] * msg[k];
			}
		}
		return val;
	}
	
	void display(){

		cerr << "act_size=" << act_set.size() << endl;
		for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
			int k = *it;
			cerr << k << ":" << y[k] << " ";
		}
		cerr << endl;
		/*for (int k = 0; k < K; k++)
			cerr << y[k] << " ";
		cerr << endl;*/
	}
};

//bigram factor
class bi_factor{
	public:
	//fixed
	int K1, K2; //number of possible labels of node l,r, resp.
	Float rho, eta;
	uni_factor* l;
	uni_factor* r;
	Float* c; // score vector: c[k1k2] = -v[k1k2/K][k1k2%K];
    pair<Float, int>* sorted_c; // sorted <score, index> vector; 
    pair<Float, int>** sorted_row; // sorted <score, index> vector of each row
    pair<Float, int>** sorted_col; // sorted <score, index> vector of each column

	//maintained
	Float* msgl; // message to uni_factor l
	Float* msgr; // messageto uni_factor r
	Float* sumcol; // sumcol[k] = sum(y[:][k])
	Float* sumrow; // sumrow[k] = sum(y[k][:])
	Float* y; // relaxed prediction matrix (vector)
	vector<int> act_set;
	bool* inside;
    int double_zero_area_index;

	inline bi_factor(uni_factor* _l, uni_factor* _r, ScoreVec* sv, Param* param){
		l = _l;
		r = _r;
		K1 = l->K;
		K2 = r->K;
		rho = param->rho;
		eta = param->eta;

		msgl = new Float[K1];
		memset(msgl, 0.0, sizeof(Float)*K1);
		msgr = new Float[K2];
		memset(msgr, 0.0, sizeof(Float)*K2);
		l->msgs.push_back(msgl);
		r->msgs.push_back(msgr);
		sumrow = new Float[K1];
		memset(sumrow, 0.0, sizeof(Float)*K1);
		sumcol = new Float[K2];
		memset(sumcol, 0.0, sizeof(Float)*K2);
		c = sv->c;
		sorted_c = sv->sorted_c;
		sorted_row = sv->sorted_row;
		sorted_col = sv->sorted_col;
        double_zero_area_index = 0;

		y = new Float[K1*K2];
		memset(y, 0.0, sizeof(Float)*K1*K2);
		
		//cache
		//grad = new Float[K1*K2];
		//memset(grad, 0.0, sizeof(Float)*K1*K2);
		inside = new bool[K1*K2];
		memset(inside, false, sizeof(bool)*K1*K2);
		act_set.clear();

		//temporary
		//fill_act_set();
	}

	~bi_factor(){

		delete[] y;
		delete[] c;
		delete[] msgl;
		delete[] msgr;
		delete[] sumcol;
		delete[] sumrow;
		//delete[] grad;
		delete[] inside;
		act_set.clear();
	}
	
	void fill_act_set(){
		act_set.clear();
		for (int kk = 0; kk < K1*K2; kk++){
			act_set.push_back(kk);
			inside[kk] = true;
		}
	}

    //should be deleted
    int best_index = -1;
    /////////
	inline void naive_search(){
		//compute gradient
	
		//find max (-gradient)
		Float gmax = -1e100;
		int max_index = -1;
		for (Int k1k2 = 0; k1k2 < K1 * K2; k1k2++){
			if (inside[k1k2]) continue;
			int k1=k1k2/K2, k2 = k1k2%K2;
			Float g = c[k1k2] + rho * (msgl[k1] + msgr[k2]);
			if (-g > gmax){
				gmax = -g;
				max_index = k1k2;
			}
		}
		if (max_index != -1){
            best_index = max_index;
            act_set.push_back(max_index);
			inside[max_index] = true;
		}
	}

    //bi_search
	inline void search(){
		//naive_search();
        //return;
        //find argmin_{k_1, k_2} gradient(k1, k2) = c[k1*K2+k2] + rho(msgl[k1] + msgr[k2])  
        int min_index = -1;
        Float gmin = 0.0;
        
        vector<int>& nnz_left = l->ever_act_set;
        vector<int>& nnz_right = r->ever_act_set;
        //area 1: msgl != 0, msgr != 0, search every entry
        for (vector<int>::iterator it_l = nnz_left.begin(); it_l != nnz_left.end(); it_l++){
            int k1 = *it_l;
            int offset = k1*K2;
            Float msgl_k1 = msgl[k1];
            for (vector<int>::iterator it_r = nnz_right.begin(); it_r != nnz_right.end(); it_r++){
                int k2 = *it_r;
                int k1k2 = offset + k2;
                if (inside[k1k2]) continue;
                Float g = c[k1k2] + rho*(msgl_k1 + msgr[k2]);
                if (g < gmin){
                    gmin = g;
                    min_index = offset + k2;
                }
            }
        }

        //area 2: msgl != 0, msgr = 0, 
        //subtask: find gmin_{k1} for each row (k1, :).
        //Here msgl_k1 is fixed, c[k1, k2] + rho*msgl_k1 + rho*msgr[k2] >= c[k1, k2] + rho*msgl_k1.
        //We visit (k1, :) in increasing order of c. Notice that:
        //1. At any time (even include (k1,k2) that is inside act_set), (c[k1, k2] + rho*msgl_k1) is a lower bound of gmin_{k1}. 
        //2. If we ever found that (!inside[k1, k2] && msgr[k2] <= 0), then (c[k1, k2] + rho*msgl_k1 + rho*msgr[k2]) can dominate this row. And we can just stop here. 
        for (vector<int>::iterator it_l = nnz_left.begin(); it_l != nnz_left.end(); it_l++){
            int k1 = *it_l;
            int offset = k1*K2;
            Float msgl_k1 = msgl[k1];
            pair<Float, int>* sorted_row_k1 = sorted_row[k1];
            //visit row (k1, :) in increasing order of c
            for (int ind = 0; ind < K2; ind++){
                int k2 = sorted_row_k1[ind].second;
                Float val = sorted_row_k1[ind].first;
                
                int k1k2 = offset + k2;
                if (inside[k1k2]){
                    //can't be used as candidate, but can provide lower bound for all feasible values
                    if (val + rho * msgl_k1 > gmin){
                        //lower bound is larger than current min value, 
                        break;
                    }
                    continue;
                }
                //try to update gmin
                if (gmin > val + rho * (msgl_k1 + msgr[k2]) ){
                    gmin = val + rho * (msgl_k1 + msgr[k2]) ;
                    min_index = offset + k2;
                }
                if (msgr[k2] <= 0){
                    //this is a dominator, since it's already updated, we simply break here.
                    break;
                }
                if (val + rho * msgl_k1 > gmin){
                    //lower bound is larger than current min value, 
                    break;
                }
            }
        }

        //area 3: msgl = 0, msgr != 0, 
        //subtask: find gmin_{k2} for each col (:, k2).
        //Here msgr_k2 is fixed, c[k1, k2] + rho*msgl[k1] + rho*msgr_k2 >= c[k1, k2] + rho*msgr_k2.
        //We visit (:, k2) in increasing order of c. Notice that:
        //1. At any time (even include (k1,k2) that is inside act_set), (c[k1, k2] + rho*msgr_k2) is a lower bound of gmin_{k2}. 
        //2. If we ever found that (!inside[k1, k2] && msgl[k1] <= 0), then (c[k1, k2] + rho*msgl[k1] + rho*msgr_k2) can dominate this row. And we can just stop here. 
        for (vector<int>::iterator it_r = nnz_right.begin(); it_r != nnz_right.end(); it_r++){
            int k2 = *it_r;
            Float msgr_k2 = msgr[k2];
            pair<Float, int>* sorted_col_k2 = sorted_col[k2];
            //visit col (:, k2) in increasing order of c
            for (int ind = 0; ind < K1; ind++){
                int k1 = sorted_col_k2[ind].second;
                Float val = sorted_col_k2[ind].first;
                int k1k2 = k1 * K2 + k2;
                if (inside[k1k2]){
                    //can't be used as candidate, but can provide lower bound for all feasible values
                    if (val + rho * msgr_k2 >= gmin){
                        //lower bound is larger than current min value, 
                        break;
                    }
                    continue;
                }
                //try to update gmin
                if (gmin > val + rho * (msgl[k1] + msgr_k2) ){
                    gmin = val + rho * (msgl[k1] + msgr_k2) ;
                    min_index = k1k2;
                }
                if (msgl[k1] <= 0){
                    //this is a dominator, since it's already updated, we simply break here.
                    break;
                }
                if (val + rho * msgr_k2 > gmin){
                    //lower bound is larger than current min value, 
                    break;
                }
            }
        }

        //area 4: msgl = 0, msgr = 0
        bool* ever_act_l = l->is_ever_act;
        bool* ever_act_r = r->is_ever_act;
        pair<Float, int> d = sorted_c[double_zero_area_index];
        while (ever_act_l[d.second / K2] || ever_act_r[d.second % K2]){
            d = sorted_c[++double_zero_area_index];
        }
        if (d.first + rho*(msgl[d.second / K2] + msgr[d.second % K2]) < gmin){
            //update gmin
            gmin = d.first + rho*(msgl[d.second / K2] + msgr[d.second % K2]);
            min_index = d.second;
        }

        //update active set
        if (min_index != -1){
            /*if (best_index != min_index){
			    cerr << "best k1=" << best_index / K2 << ", k2=" << best_index % K2 << ", c=" << c[best_index] << ", msgl=" << rho*msgl[best_index / K2] << ", msgr=" << rho*msgr[best_index % K2] << endl;
                cerr << "found k1=" << min_index / K2 << ", k2=" << min_index % K2 << ", c=" << c[min_index] << ", msgl=" << rho*msgl[min_index / K2] << ", msgr=" << rho*msgr[min_index % K2] << endl;
                assert(fabs(gmin-c[best_index] - rho*(msgl[best_index / K2] + msgr[best_index % K2]) ) < 1e-6);
            }*/
            //adding min_index into active set, also update ever_act_set for l and r node
            act_set.push_back(min_index);
            inside[min_index] = true;
            if (!(l->is_ever_act[min_index / K2])){
                l->is_ever_act[min_index / K2] = true;
                l->ever_act_set.push_back(min_index / K2);
            }
            if (!(r->is_ever_act[min_index % K2])){
                r->is_ever_act[min_index % K2] = true;
                r->ever_act_set.push_back(min_index % K2);
            }
        }
	}

	//       min_Y  <Y, -v> + \rho/2 ( \| msgl \|_2^2 + \| msgr \|_2^2 )
	// <===> min_Y  A/2 \|Y\|_2^2 + <B,  Y>
	// <===> min_Y \| Y - (-B/A) \|_2^2 
	//        s.t. 	A = 2 * \rho * K (?)
	// 		B_{(k_1, k_2)} = -v[k1][k2] + \rho * (msgl[k_1] - sumrow[k1] + msgr[k_2] - sumcol[k2])
	// 		0 <= Y <= 1
	//	Let C := -B/A
	inline void subsolve(){
		/*
		int* sl = new int[K1];
		int* sr = new int[K2];
		memset(sl, 0, sizeof(int)*K1);
		memset(sr, 0, sizeof(int)*K2);
		int max_sl = 0, max_sr = 0;
		for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
			int k1k2 = *it;
			int k1 = k1k2 / K2, k2 = k1k2 % K2;
			sl[k1]++; sr[k2]++;
			if (max_sl < sl[k1])
				max_sl = sl[k1];
			if (max_sr < sr[k2])
				max_sr = sr[k2];
		}
		*/
		
		Float A = rho * (K1 + K2); //(l->act_set.size() + r->act_set.size());
		Float* y_new = new Float[act_set.size()];

		int act_count = 0;
		
		if (fabs(rho) < 1e-12){
			//min_y <c, y>
			Float cmin = 1e300;
			int min_index = -1;
			act_count = 0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k = *it;
				if (c[k] < cmin){
					cmin = c[k];
					min_index = act_count;
				}
			}
			assert(min_index != -1 && "smallest coordinate should exist");
			
			memset(y_new, 0.0, sizeof(Float)*act_set.size());
			y_new[min_index] = 1.0;
		} else {
			Float* C = new Float[act_set.size()];
			act_count = 0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k1k2 = *it;
				int k1 = k1k2 / K2;
				int k2 = k1k2 % K2;
				C[act_count] = -(c[k1k2] + rho * (msgl[k1] + msgr[k2] - sumrow[k1] - sumcol[k2]) - A*y[k1k2])/A; // C = -B/A
			}
			//cerr << "before subsolve, val=" << func_val() << endl;

			solve_simplex(act_set.size(), y_new, C);
			/*act_count = 0;
			  for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
			  int k1k2 = *it;
			  int k1 = k1k2 / K;
			  int k2 = k1k2 % K;
			  cerr << "(" << k1 << "," << k2 << "):" << C[act_count] << "=" << (-c[k1k2]/A) << "-" << (rho/A*(msgl[k1])) << "-" << (rho/A*(msgr[k2])) << "+" << (rho/A*(sumrow[k1])) << "+" << (rho/A*(sumcol[k2])) << " ";
			  }
			  cerr << endl;*/
			delete[] C;
		}
		
		//update y and message
		act_count = 0;
		for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
			int k1k2 = *it;
			if (fabs(y_new[act_count]) < 1e-12){
				y_new[act_count] = 0.0;
			}
			Float delta_y = y_new[act_count] - y[k1k2];
			int k1 = k1k2 / K2, k2 = k1k2 % K2;
			msgl[k1] += delta_y; sumrow[k1] += delta_y;
			msgr[k2] += delta_y; sumcol[k2] += delta_y;
		}
	
		vector<int> next_act_set;
		next_act_set.clear();
		act_count = 0;
		for(vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
			int k1k2 = *it;
			y[k1k2] = y_new[act_count];
			//possible shrink here
			if (y[k1k2] > 1e-12){
				next_act_set.push_back(k1k2);
			} else {
				inside[k1k2] = false;
			}
		}

		//some check here
		/*act_count = 0;
		for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
			int k1k2 = *it;
			int k1 = k1k2 / K;
			int k2 = k1k2 % K;
			C[act_count] = -(-v[k1][k2] + rho * (msgl[k1] + msgr[k2] - sumrow[k1] - sumcol[k2]))/A; // C = -B/A
		}
		solve_simplex(act_set.size(), y_new, C);
		act_count = 0;
		for(vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
			int k1k2 = *it;
			assert(fabs(y[k1k2] - y_new[act_count]) < 1e-6);
		}*/
		///////////////////////////

		act_set = next_act_set;
		
		//cerr << "after subsolve, val=" << func_val() << endl;

		delete[] y_new;
	}
	
	//     \mu^{t+1} - \mu^t
	//   = \msg^{t+1} - \msg^t
	//   = eta * ( (\sum_j y[:][j]) - y[:])
	inline void update_multipliers(){
		//update msgl, msgr
		Float* y_l = l->y;
		Float* y_r = r->y;
		for (int k = 0; k < K1; k++){
			msgl[k] += eta * (sumrow[k] - y_l[k]);
		}
		for (int k = 0; k < K2; k++){
			msgr[k] += eta * (sumcol[k] - y_r[k]);
		}
	}

    inline int nnz_msg(){
        int nnz = 0;
        for (int i = 0; i < K1; i++){
            if (msgl[i] < 0)
                nnz++;
        }
        for (int i = 0; i < K2; i++){
            if (msgr[i] < 0)
                nnz++;
        }
        return nnz;
    }

	Float score(){
		Float score = 0.0;
		for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
			int k1k2 = *it;
			score += c[k1k2]*y[k1k2];
		}
		return score;
	}
	
	// F = c^T Y + \rho/2 (\|msgl \|^2 + \|msgr\|^2_2)
	Float func_val(){
		Float val = 0.0;
		for (int k = 0; k < K1*K2; k++){
			val += c[k]*y[k];
		}
		for (int k = 0; k < K1; k++){
			val += rho/2 * msgl[k] * msgl[k];
		}
		for (int k = 0; k < K2; k++){
			val += rho/2 * msgr[k] * msgr[k];
		}
		return val;
	}
	
	Float infea(){
		Float p_inf = 0;
		Float* y_l = l->y;
		Float* y_r = r->y;
		for (int k = 0; k < K1; k++){
			p_inf += fabs(sumrow[k] - y_l[k]);
		}
		for (int k = 0; k < K2; k++){
			p_inf += fabs(sumcol[k] - y_r[k]);
		}
		return p_inf;
	}
	
	void display(){
		/*for (int k = 0; k < K*K; k++){
			if (k % K == 0)
				cerr << endl;
			cerr << y[k] << " ";
		}
		cerr << endl;
		cerr << endl;*/
		//cerr << "norm(c, 2)=" << norm_sq(c, K*K) << endl;
		for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
			int kk = *it;
			cerr << "(" << (kk/K2) << "," << (kk%K2) << ")" << ":" << y[kk] << " ";
		}
		cerr << endl;
	}

};
