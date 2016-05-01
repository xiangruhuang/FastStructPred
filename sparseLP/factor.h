//prediction

#include "util.h"

class Stats{
	public:
		int num_bi;
		int area1;
		int area23;
		int area4;
		Float bi_act_size;

		int num_uni; 
		Float uni_act_size;
		Float uni_ever_act_size;

		double uni_search_time = 0.0;
		double uni_subsolve_time = 0.0;
		double bi_search_time = 0.0;
		double bi_subsolve_time = 0.0;
		double maintain_time = 0.0;
		double construct_time = 0.0;

		Stats(){
			clear();
			uni_search_time = 0.0;
			uni_subsolve_time = 0.0;
			bi_search_time = 0.0;
			bi_subsolve_time = 0.0;
			maintain_time = 0.0;
			construct_time = 0.0;
		}
		void display(){
			cerr << ", area1=" << (double)area1/num_bi << ", area23=" << (double)area23/num_bi
				<< ", area4=" << (double)area4/num_bi
				<< ", bi_act_size=" << (double)bi_act_size/num_bi; 

			cerr << ", uni_act_size=" << (double)uni_act_size/num_uni
				<< ", uni_ever_act_size=" << (double)uni_ever_act_size/num_uni;
		}
		void clear(){
			num_bi = 0; 
			area1 = 0; area23 = 0; area4 = 0; bi_act_size = 0;

			num_uni = 0;
			uni_act_size = 0; uni_ever_act_size = 0; 
		}
};

Stats* stats = new Stats();

//unigram factor, y follows simplex constraints
class uni_factor{
	public:
		//fixed
		int K;
		Float rho;
		Float* c; // score vector, c[k] = -<w_k, x>
        Float nnz_tol = 1e-6;

		//maintained
		Float* grad;
		Float* y;
		bool* inside;
		vector<Float*> msgs;
		vector<int> act_set;
		vector<int> ever_act_set;
		bool* is_ever_act;
		int searched_index;

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
			//ever_act_set.clear();
			for (int k = 0; k < K; k++){
				act_set.push_back(k);
				//ever_act_set.push_back(k);
				//is_ever_act[k] = true;
				inside[k] = true;
			}
		}

		//uni_search
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

			searched_index = max_index;
			if (max_index != -1){
				act_set.push_back(max_index);
				inside[max_index] = true;
			}
		}


		//	min_{y \in simplex} <c, y> + \rho/2 \sum_{msg \in msgs} \| (msg + y) - y \|_2^2
		// <===>min_{y \in simplex} \| y - 1/|msgs| ( \sum_{msg \in msgs} (msg + y) - 1/\rho c ) \|_2^2
		// <===>min_{y \in simplex} \| y - b \|_2^2
		// uni_subsolve
		inline void subsolve(){
			if (act_set.size() == 0)
				return;
			Float* y_new = new Float[act_set.size()];
			int act_count = 0;
			if (msgs.size() == 0 || fabs(rho) < 1e-12){
				//min_y <c, y>
				cerr << "degenerated uni_subsolve!!!!!" << endl;
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
				delete[] b;
			}

			for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
				Float* msg = *m;
				act_count = 0;
				if (fabs(y_new[act_count]) < nnz_tol){
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
				if (y[k] >= nnz_tol ){
						//this index is justed added by search()
						//Only if this index is active after subsolve, then it's added to ever_act_set
                    if (!is_ever_act[k] && k == searched_index){
                        is_ever_act[k] = true;
                        ever_act_set.push_back(k);
                    }
					next_act_set.push_back(k);
				} else {
					inside[k] = false;
				}
			}
			act_set = next_act_set;

			delete[] y_new;
		}

		//goal: minimize score
		inline Float score(){
			Float score = 0.0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				score += c[k]*y[k];
			}
			return score;
		}

		// F = c^T y + \rho/2 \sum_{msg} \|msg \|^2
		inline Float func_val(){
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

		inline void display(){

			cerr << endl;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				cerr << k << ":" << y[k] << " ";
			}
			cerr << endl;
			/*for (int k = 0; k < K; k++)
			  cerr << y[k] << " ";
			  cerr << endl;*/
		}

        int dinf_index = -1;
        inline Float dual_inf(){
            for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
                int k = *it;
                grad[k] = c[k];
            }
            for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
                Float* msg = *m;
                for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
                    int k = *it;
                    grad[k] -= rho * msg[k];
                }
            }
            //max gradient inside active set
            Float gmax = 0.0;
            dinf_index = -1;
            for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
                int k = *it;
                if (-grad[k] > gmax && (y[k] < 1 - nnz_tol) ){
                    gmax = -grad[k];
                    dinf_index = k;
                }
                if (grad[k] > gmax && (y[k] > nnz_tol) ){
                    gmax = grad[k];
                    dinf_index = k;
                }
            }
            return gmax;
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
        Float nnz_tol = 1e-6;
		//maintained
		Float* msgl; // message to uni_factor l
		Float* msgr; // messageto uni_factor r
		Float* sumcol; // sumcol[k] = sum(y[:][k])
		Float* sumrow; // sumrow[k] = sum(y[k][:])
		//Float* Y; // relaxed prediction matrix (vector)
		vector<pair<int, Float>> act_set; // <index, val> pair
		bool* inside;
		int double_zero_area_index;
		int searched_index;

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

			//Y = new Float[K1*K2];
			//memset(Y, 0.0, sizeof(Float)*K1*K2);

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

			//delete[] Y;
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
				act_set.push_back(make_pair(kk, 0.0));
				inside[kk] = true;
			}
		}

        Float gmax = -1e100;
		inline void naive_search(){
			//compute gradient

			//find max (-gradient)
			gmax = -1e100;
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
			searched_index = max_index;
			if (max_index != -1){
				act_set.push_back(make_pair(max_index, 0.0));
				inside[max_index] = true;
			}
            //cerr << "gmax=" << gmax << ", searched_index=" << searched_index << ", ever_l="<< l->is_ever_act[searched_index / K2] << ", ever_r=" << r->is_ever_act[searched_index % K2]
              //  << ", c=" << c[searched_index] << ", msgl=" << msgl[searched_index / K2] << ", msgr=" << msgr[searched_index % K2] << endl;
		}

		//bi_search()
		inline void search(){
			//naive_search();
			//return;
            //find argmin_{k_1, k_2} gradient(k1, k2) = c[k1*K2+k2] + rho(msgl[k1] + msgr[k2])  
			int min_index = -1;
			Float gmin = 0.0;

			vector<int>& nnz_left = l->ever_act_set;
			vector<int>& nnz_right = r->ever_act_set;
            int nnz_l = nnz_left.size();
            int nnz_r = nnz_right.size();
			//maintain stats of search 
            int* index_l = new int[nnz_l];
            int* index_r = new int[nnz_r];
            for (int i = 0; i < nnz_l; i++){
                index_l[i] = nnz_left[i];
            }
            for (int i = 0; i < nnz_r; i++){
                index_r[i] = nnz_right[i];
            }
            //sorted in decreasing order
            sort(index_l, index_l+nnz_l, ScoreComp(msgl));
            sort(index_r, index_r+nnz_r, ScoreComp(msgr));
            int* record_l = new int[K1];
            for (int i = 0; i < K1; i++)
                record_l[i] = -1;

			int num = 10*(nnz_left.size() + nnz_right.size());
            int minl = K1, minr = K2;
            for (int i = 0; i < num; i++){
                if (i >= K1 * K2) break;
                pair<Float, int> p = sorted_c[i];
                int k1k2 = p.second;
                if (inside[k1k2]) continue;
                int k1 = k1k2 / K2, k2 = k1k2 % K2;
                Float g = p.first + rho * (msgl[k1] + msgr[k2]);
                if (g < gmin){
                    gmin = g;
                    min_index = k1k2;
                }
                if (record_l[k1] == -1 || msgr[record_l[k1]] > msgr[k2]){
                    record_l[k1] = k2;
                }
            }
            /*for (int i = 0; i < nnz_l; i++)
                cerr << index_l[i] << ":" << msgl[index_l[i]] << " ";
            cerr << endl;
            for (int i = 0; i < nnz_r; i++)
                cerr << index_r[i] << ":" << msgr[index_r[i]] << " ";
            cerr << endl;*/
            ////////
            Float c_lower =  sorted_c[num-1].first;
            /*if (c_lower + rho*(msgl[index_l[nnz_l-1]] + msgr[index_r[nnz_r-1]]) >= gmin){
                assert(fabs(gmin+gmax) < 1e-6);
                if (min_index != -1){
                    //adding min_index into active set, update of ever_act_set is done after subsolve
                    act_set.push_back(make_pair(min_index, 0.0));
                    inside[min_index] = true;
                    return;
                }
            }*/
            //cerr << c_lower << "/" << gmin << endl;
			//area 1: msgl != 0, msgr != 0, search every entry
            int limit = index_r[0];
			for (int il = nnz_l-1; il >= 0; il--){
				int k1 = index_l[il];
				int offset = k1*K2;
                Float msgl_k1 = msgl[k1];
                if (record_l[k1] != -1 && msgr[limit] > msgr[record_l[k1]]){
                    limit = record_l[k1];
                }
                //cerr << "k1=" << k1 << ", limit=" << limit << endl;
                if (limit == index_r[nnz_l-1])
                    break;
				for (int ir = nnz_r-1; ir >= 0; ir--){
			        stats->area1++;
                    int k2 = index_r[ir];
                    if (k2 == limit)
                        break;
					int k1k2 = offset + k2;
					if (inside[k1k2]) continue;
                    Float g = c[k1k2] + rho*(msgl_k1 + msgr[k2]);
                    if (g < gmin){
                        gmin = g;
                        min_index = offset + k2;
                    }
				}
			}

            delete[] index_l;
            delete[] index_r;
            delete[] record_l;

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

					//maintain stats of search 
					stats->area23++;
					////////
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
					//maintain stats of search 
					stats->area23++;
					////////
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
            if (double_zero_area_index < K1*K2){
			    pair<Float, int> d = sorted_c[double_zero_area_index];
                //maintain stats of search 
                stats->area4++;
                ////////
                while (double_zero_area_index < K1*K2 && (ever_act_l[d.second / K2] || ever_act_r[d.second % K2])){
                    //maintain stats of search
                    stats->area4++;
                    ////////
                    d = sorted_c[++double_zero_area_index];
                }
                if (double_zero_area_index < K1*K2 && (d.first + rho*(msgl[d.second / K2] + msgr[d.second % K2]) < gmin)){
                    //update gmin
                    gmin = d.first + rho*(msgl[d.second / K2] + msgr[d.second % K2]);
                    min_index = d.second;
                }
            }

			//update active set
			if (min_index != -1){
				/*if (searched_index != min_index){
				  cerr << "best k1=" << searched_index / K2 << ", k2=" << searched_index % K2 << ", c=" << c[searched_index] << ", msgl=" << rho*msgl[searched_index / K2] << ", msgr=" << rho*msgr[searched_index % K2] << endl;
				  cerr << "found k1=" << min_index / K2 << ", k2=" << min_index % K2 << ", c=" << c[min_index] << ", msgl=" << rho*msgl[min_index / K2] << ", msgr=" << rho*msgr[min_index % K2] << endl;
				  assert(fabs(gmin-c[searched_index] - rho*(msgl[searched_index / K2] + msgr[searched_index % K2]) ) < 1e-6);
				  }*/
				//adding min_index into active set, update of ever_act_set is done after subsolve
				act_set.push_back(make_pair(min_index, 0.0));
				inside[min_index] = true;
			}
            //cerr << "gmin=" << gmin << ", min_index=" << min_index << ", ever_l="<< l->is_ever_act[min_index / K2] << ", ever_r=" << r->is_ever_act[min_index % K2]
            //    << ", c=" << c[min_index]<< ", msgl=" << msgl[min_index / K2] << ", msgr=" << msgr[min_index % K2] << endl;
            //assert(fabs(gmin+gmax) < 1e-6);
		}

		//       min_Y  <Y, -v> + \rho/2 ( \| msgl \|_2^2 + \| msgr \|_2^2 )
		// <===> min_Y  A/2 \|Y\|_2^2 + <B,  Y>
		// <===> min_Y \| Y - (-B/A) \|_2^2 
		//        s.t. 	A = 2 * \rho * K (?)
		// 		B_{(k_1, k_2)} = -v[k1][k2] + \rho * (msgl[k_1] - sumrow[k1] + msgr[k_2] - sumcol[k2])
		// 		0 <= Y <= 1
		//	Let C := -B/A
		//	bi_subsolve()
		inline void subsolve(){

			Float A = rho * (2 + l->ever_act_set.size() + r->ever_act_set.size());
			Float* Y_new = new Float[act_set.size()];

			int act_count = 0;

			if (fabs(A) < 1e-12 || act_set.size() == 0){
				//min_Y <c, Y> since message has weight zero
				cerr << "degenerated bi_subsolve!!!!!" << endl;
                Float cmin = 1e300;
				int min_index = -1;
				act_count = 0;
				for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
					int k = it->first;
					if (c[k] < cmin){
						cmin = c[k];
						min_index = act_count;
					}
				}
				assert(min_index != -1 && "smallest coordinate should exist");

				memset(Y_new, 0.0, sizeof(Float)*act_set.size());
				Y_new[min_index] = 1.0;
			} else {
				Float* C = new Float[act_set.size()];
				act_count = 0;
				for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
					int k1k2 = it->first;
					int k1 = k1k2 / K2;
					int k2 = k1k2 % K2;
					C[act_count] = -(c[k1k2] + rho * (msgl[k1] + msgr[k2] - sumrow[k1] - sumcol[k2]))/A + it->second; // C = -B/A
				}
				solve_simplex(act_set.size(), Y_new, C);
				delete[] C;
			}

			//update Y and message
			act_count = 0;
			for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k1k2 = it->first;
				if (fabs(Y_new[act_count]) < nnz_tol){
					Y_new[act_count] = 0.0;
				}
				Float delta_Y = Y_new[act_count] - it->second;
				int k1 = k1k2 / K2, k2 = k1k2 % K2;
				msgl[k1] += delta_Y; sumrow[k1] += delta_Y;
				msgr[k2] += delta_Y; sumcol[k2] += delta_Y;
			}

			vector<pair<int, Float>> next_act_set;
			next_act_set.clear();
			act_count = 0;
			for(vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k1k2 = it->first;
				it->second = Y_new[act_count];
				//possible shrink here
				if (it->second > nnz_tol){
                    int k1 = k1k2 / K2, k2 = k1k2 % K2;
                    if (!l->is_ever_act[k1]){
                        l->is_ever_act[k1] = true;
                        l->ever_act_set.push_back(k1);
                    }
                    if (!r->is_ever_act[k2]){
                        r->is_ever_act[k2] = true;
                        r->ever_act_set.push_back(k2);
                    }
					next_act_set.push_back(make_pair(k1k2, it->second));
				} else {
					inside[k1k2] = false;
				}
			}

			act_set = next_act_set;

			delete[] Y_new;
		}

		//     \mu^{t+1} - \mu^t
		//   = \msg^{t+1} - \msg^t
		//   = eta * ( (\sum_j y[:][j]) - y[:])
		inline void update_multipliers(){
			//update msgl, msgr
			Float* y_l = l->y;
			Float* y_r = r->y;
			for (int k = 0; k < K1; k++){
				msgl[k] = msgl[k] + eta * (sumrow[k] - y_l[k]);
			}
			for (int k = 0; k < K2; k++){
				msgr[k] = msgr[k] + eta * (sumcol[k] - y_r[k]);
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

		inline Float score(){
			Float score = 0.0;
			for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k1k2 = it->first;
				score += c[k1k2]*it->second;
			}
			return score;
		}

		// F = c^T Y + \rho/2 (\|msgl \|^2 + \|msgr\|^2_2)
		inline Float func_val(){
			Float val = 0.0;
			for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				val += c[it->first]*it->second;
			}
			for (int k = 0; k < K1; k++){
				val += rho/2 * msgl[k] * msgl[k];
			}
			for (int k = 0; k < K2; k++){
				val += rho/2 * msgr[k] * msgr[k];
			}
			return val;
		}

		inline Float infea(){
			Float p_inf = 0.0;
			Float* y_l = l->y;
			Float* y_r = r->y;
			for (int k = 0; k < K1; k++){
				Float inf = fabs(sumrow[k] - y_l[k]);
                if (inf > p_inf)
                    p_inf = inf;
			}
			for (int k = 0; k < K2; k++){
				Float inf = fabs(sumcol[k] - y_r[k]);
                if (inf > p_inf)
                    p_inf = inf;
			}
			return p_inf;
		}

		inline void display(){
			cerr << endl;
            for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int kk = it->first;
				cerr << "(" << (kk/K2) << "," << (kk%K2) << ")" << ":" << it->second << " ";
			}
			cerr << endl;
		}
        
        int dinf_index = -1;
        inline Float dual_inf(){
            Float gmax = 0.0;
            dinf_index = -1;
			for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k1k2 = it->first;
                int k1 = k1k2/K2, k2 = k1k2 % K2;
                Float g = c[k1k2] + rho*(msgl[k1] + msgr[k2]);
                if (-g > gmax && (it->second < 1 - nnz_tol)){
                    gmax = -g;
                    dinf_index = k1k2;
                }
                if (g > gmax && it->second > nnz_tol){
                    gmax = g;
                    dinf_index = k1k2;
                }
            }
            return gmax;
        }
};

