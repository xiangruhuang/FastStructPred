#ifndef FACTOR_H
#define FACTOR_H
//prediction

#include "util.h"

class Stats{
	public:
		int num_bi;
		Float area1;
		int area23;
		int area4;
		Float bi_act_size;

		int num_uni; 
		Float uni_act_size;
		Float ever_nnz_msg_size;
		Float delta_Y_l1;
		Float weight_b;		

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
			cerr << ", uni_act_size=" << (double)uni_act_size/num_uni;

			cerr << ", area1=" << (double)area1/num_bi << ", area23=" << (double)area23/num_bi
				<< ", area4=" << (double)area4/num_bi
				<< ", bi_act_size=" << (double)bi_act_size/num_bi 
				<< ", bi_ever_nnz_msg=" << (double)ever_nnz_msg_size/num_bi;
		}

		void clear(){
			num_bi = 0; 
			area1 = 0; area23 = 0; area4 = 0; bi_act_size = 0;

			num_uni = 0;
			uni_act_size = 0; ever_nnz_msg_size = 0; 
		}

		void display_time(){
			cerr << ", uni_search=" << uni_search_time
				<< ", uni_subsolve=" << uni_subsolve_time
				<< ", bi_search=" << bi_search_time
				<< ", bi_subsolve=" << bi_subsolve_time 
				<< ", maintain=" << maintain_time 
				<< ", construct=" << construct_time;
		}
};

Stats* stats = new Stats();

class Factor{
	public:
		virtual inline void search(){
		}
		virtual inline void subsolve(){
		}
};

class BiFactor;

//unigram factor, y follows simplex constraints
class UniFactor : public Factor{
	public:
		//fixed
		int K;
		Float rho;
		Float* c; // score vector, c[k] = -<w_k, x>
		Float nnz_tol;

		bool shrink;
		bool soft_bcfw;

		//maintained
		Float* grad;
		Float* y;
		bool* inside;
		vector<Float*> msgs;
		vector<int> act_set;
		//vector<int> ever_act_set;
		//bool* is_ever_act;
		int searched_index;
		vector<BiFactor*> edge_to_right;
		vector<BiFactor*> edge_to_left;
		Float largest, smallest;
		Float weight_b = 1.0;

		inline UniFactor(int _K, Float* _c, Param* param){
			K = _K;
			rho = param->rho;
			nnz_tol = param->nnz_tol;
			//compute score vector
			c = _c;
			largest = -1e100;
			smallest = 1e100;
			for (int i = 0; i < K; i++){
				if (fabs(c[i]) > 1e5){
					continue;
				}
				if (largest < c[i]){
					largest = c[i];
				}
				if (smallest > c[i]){
					smallest = c[i];
				}
			}

			//cache of gradient
			grad = new Float[K];
			memset(grad, 0.0, sizeof(Float)*K);

			//relaxed prediction vector
			y = new Float[K];
			memset(y, 0.0, sizeof(Float)*K);

			inside = new bool[K];
			memset(inside, false, sizeof(bool)*K);
			act_set.clear();
			//ever_act_set.clear();
			edge_to_right.clear();
			edge_to_left.clear();
			//is_ever_act = new bool[K];
			//memset(is_ever_act, false, sizeof(bool)*K);
			msgs.clear();

			shrink = true;
			//fill_act_set(); shrink = false;
			soft_bcfw = false;
			if (param->solver == 3){
				shrink = false;
				soft_bcfw = true;
				fill_act_set();
			}
		}

		~UniFactor(){
			delete[] y;
			delete[] grad;
			delete[] inside;
			//delete[] is_ever_act;
			act_set.clear();
			msgs.clear();
			edge_to_right.clear();
			edge_to_left.clear();
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

		inline void adding_ever_act(int k);

		//uni_search()
		inline void search(){
			stats->uni_search_time -= get_current_time();
			//compute gradient of y_i
			for (int k = 0; k < K; k++){
				grad[k] = c[k];
			}
			for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
				Float* msg = *m;
				for (int k = 0; k < K; k++)
					grad[k] -= rho * msg[k];
			}
			Float gmax = -1e100;
			int max_index = -1;
			for (int k = 0; k < K; k++){
				if (inside[k]) continue;
				//if not inside, y_k is guaranteed to be zero, and y_k is nonnegative, so we only care gradient < 0
				//if (grad[k] > 0 && act_set.size() != 0) continue;
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
			stats->uni_search_time += get_current_time();
		}


		//	min_{y \in simplex} <c, y> + \rho/2 \sum_{msg \in msgs} \| (msg + y) - y \|_2^2
		// <===>min_{y \in simplex} \| y - 1/|msgs| ( \sum_{msg \in msgs} (msg + y) - 1/\rho c ) \|_2^2
		// <===>min_{y \in simplex} \| y - b \|_2^2
		// uni_subsolve()
		//bool output = false;
		inline void subsolve(){
			if (act_set.size() == 0)
				return;
			stats->uni_subsolve_time -= get_current_time();
			Float* y_new = new Float[act_set.size()];
			int act_count = 0;
			if (soft_bcfw){
				Float l1_sum = 0.0;
				for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
					int k = *it;
					grad[k] = c[k];
					l1_sum += y[k];
				}
				if (l1_sum < 1.0){
					int min_cindex = 0;
					for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
						y_new[*it] = 0.0;
						if (c[*it] < c[min_cindex]){
							min_cindex = *it;
						}
					}
					y_new[min_cindex] = 1.0;
				} else {
					Float dot_prod = 0.0;
					for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
						Float* msg = *m;
						for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
							int k = *it;
							grad[k] -= rho * msg[k];
						}
					}
					//max gradient inside active set
					Float gmin = 1e300;
					int min_index = -1;
					for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
						int k = *it;
						dot_prod += y[k] * grad[k];
						if (grad[k] < gmin){
							gmin = grad[k];
							min_index = k;
						}
					}
					//cerr << "get direction:" << min_index << endl;
					if (dot_prod - gmin <= 1e-12){
						delete[] y_new;
						stats->uni_subsolve_time += get_current_time();
						return;
					}

					//up = -<grad, \Delta Y>, down = A/2 \|\Delta Y\|_2^2
					Float up = 0.0, down = 0.0;
					Float A = msgs.size() * rho;

					for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
						int k = *it;
						Float g = grad[k];
						Float delta_y = -y[k];
						if (k == min_index){
							delta_y += 1;
						}
						up -= g * delta_y;
						down += A * delta_y * delta_y;
					}
					//cerr << "up=" << up << ", down=" << down << endl;
					//assert(fabs(down)>1e-12);
					Float gamma = up/down;
					if (gamma < 0)
						gamma = 0;
					if (gamma > 1)
						gamma = 1;
					//cerr << "gamma=" << gamma << endl;
					act_count = 0;
					for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
						int k = *it;
						Float delta_y = -y[k];
						if (k == min_index){
							delta_y += 1;
						}
						y_new[act_count] = y[k] + delta_y * gamma;
					}
				}
			} else {
				if (msgs.size() == 0 || fabs(rho) < 1e-12){
					//min_y <c, y>
					//cerr << "degenerated uni_subsolve!!!!!" << endl;
					Float cmin = 1e300;
					int min_index = -1;
					act_count = 0;
					for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
						int k = *it;
						//cerr << k << ":" << c[k] << endl;
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
						//cerr << "msg_i:\t";
						for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
							int k = *it;
							b[act_count] += msg[k];
							//cerr << k << ":" << msg[k] << " ";
						}
						//cerr << endl;
					}
					int n = msgs.size();
					act_count = 0;
					weight_b = 0.0;
					for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
						int k = *it;
						b[act_count] -= c[k]/rho;
						b[act_count] /= (Float)n;
						b[act_count] += y[k];
						if (b[act_count] > 0.0){
							weight_b += b[act_count];
						}
						//cerr << b[act_count] << " ";
					}
					stats->weight_b += weight_b;
					//cerr << endl;
					solve_simplex(act_set.size(), y_new, b);
					delete[] b;
				}
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
					stats->delta_Y_l1 += fabs(delta_y);
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
				if (!shrink || y[k] >= nnz_tol ){
					//this index is justed added by search()
					//Only if this index is active after subsolve, then it's added to ever_act_set
					if (k == searched_index){
						adding_ever_act(k);
					}
					next_act_set.push_back(k);
				} else {
					inside[k] = false;
				}
			}
			act_set = next_act_set;

			delete[] y_new;
			stats->uni_subsolve_time += get_current_time();
		}

		int recent_pred = -1;
		//goal: minimize score
		inline Float score(){
			/*Float score = 0.0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				score += c[k]*y[k];
			}
			return score;
			*/
			Float max_y = -1;
			recent_pred = -1;
			//randomly select when there is a tie
			random_shuffle(act_set.begin(), act_set.end());
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				if (y[k] > max_y){
					recent_pred = k;
					max_y = y[k];
				}
			}
			//cerr << "recent_pred=" << recent_pred << ", c=" << c[recent_pred] << endl;
			return c[recent_pred];
		}
		
        inline Float rel_score(){
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

			//cerr << grad[0] << " " << grad[1] << endl;
			cerr << endl;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				cerr << k << ":" << y[k] << ":" << c[k] << " ";
			}
			cerr << endl;
			/*for (int k = 0; k < K; k++)
			  cerr << y[k] << " ";
			  cerr << endl;*/
		}

		int dinf_index = -1;
		//uni_dual_inf()
		inline Float dual_inf(){
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				grad[k] = c[k];
			}
			Float dot_prod = 0.0;
			for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
				Float* msg = *m;
				for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
					int k = *it;
					grad[k] -= rho * msg[k];
				}
			}
			//max gradient inside active set
			Float gmin = 1e300;
			dinf_index = -1;
			int min_index = -1;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				dot_prod += y[k] * grad[k];
				if (grad[k] < gmin){
					gmin = grad[k];
					min_index = k;
				}
			}
			if (dot_prod - gmin > 0){
				dinf_index = min_index;
			}
			return dot_prod - gmin;
		}
};

//bigram factor
class BiFactor : public Factor{
	public:
		//fixed
		int K1, K2; //number of possible labels of node l,r, resp.
		Float rho, eta;
		UniFactor* l;
		UniFactor* r;
		Float* c; // score vector: c[k1k2] = -v[k1k2/K][k1k2%K];
		pair<Float, int>* sorted_c; // sorted <score, index> vector; 
		pair<Float, int>** sorted_row; // sorted <score, index> vector of each row
		pair<Float, int>** sorted_col; // sorted <score, index> vector of each column
		Float nnz_tol;
		bool soft_bcfw;
		bool shrink;
		//maintained
		Float* msgl; // message to UniFactor l
		Float* msgr; // message to UniFactor r
		Float* sumcol; // sumcol[k] = sum(y[:][k])
		Float* sumrow; // sumrow[k] = sum(y[k][:])
		//Float* Y; // relaxed prediction matrix (vector)
		vector<pair<int, Float>> act_set; // <index, val> pair
		vector<int> ever_nnz_msg_l; // <val, index> pair (easier to sort)
		vector<int> ever_nnz_msg_r; // <val, index> pair (easier to sort)
		bool* is_ever_nnz_l;
		bool* is_ever_nnz_r;
		static bool* inside;
		int double_zero_area_index;
		int searched_index;
		int check_num;
		vector<pair<Float, int>> sorted_ever_act_c;
		bool updated;
		Float weight_b = 1.0;
		//Float smallest, largest;
		Float* last_msg_hat_l;
		Float* last_msg_hat_r;
		bool accelerate = false;

		inline BiFactor(UniFactor* _l, UniFactor* _r, ScoreVec* sv, Param* param){
			l = _l;
			r = _r;
			K1 = l->K;
			K2 = r->K;
			rho = param->rho;
			nnz_tol = param->nnz_tol;
			
			if (accelerate){
				last_msg_hat_l = new Float[K1];
				memset(last_msg_hat_l, 0.0, sizeof(Float)*K1);
				last_msg_hat_r = new Float[K2];
				memset(last_msg_hat_r, 0.0, sizeof(Float)*K2);
			}

			msgl = new Float[K1];
			memset(msgl, 0.0, sizeof(Float)*K1);
			msgr = new Float[K2];
			memset(msgr, 0.0, sizeof(Float)*K2);
			l->msgs.push_back(msgl);
			r->msgs.push_back(msgr);
			l->edge_to_right.push_back((BiFactor*)this);
			r->edge_to_left.push_back((BiFactor*)this);
			updated = false, check_num = (K1+K2)/2;
			sumrow = new Float[K1];
			memset(sumrow, 0.0, sizeof(Float)*K1);
			sumcol = new Float[K2];
			memset(sumcol, 0.0, sizeof(Float)*K2);
			c = sv->c;
			sorted_c = sv->sorted_c;
			sorted_row = sv->sorted_row;
			sorted_col = sv->sorted_col;
			double_zero_area_index = 0;
			eta = param->eta;

			//Y = new Float[K1*K2];
			//memset(Y, 0.0, sizeof(Float)*K1*K2);

			//cache
			//grad = new Float[K1*K2];
			//memset(grad, 0.0, sizeof(Float)*K1*K2);
			if (inside == NULL){
				inside = new bool[K1*K2*5];
				memset(inside, false, sizeof(bool)*K1*K2*5);
			}
			act_set.clear();
			sorted_ever_act_c.clear();

			ever_nnz_msg_l.clear();
			is_ever_nnz_l = new bool[K1]; 
			memset(is_ever_nnz_l, false, sizeof(bool)*K1);
			ever_nnz_msg_r.clear();
			is_ever_nnz_r = new bool[K2];
			memset(is_ever_nnz_r, false, sizeof(bool)*K2);
			//temporary
            		shrink = true;
			//fill_act_set(); shrink=false;
			soft_bcfw = false;
			if (param->solver==3){
				shrink = false;
				soft_bcfw = true;
				fill_act_set();
			}
		}

		~BiFactor(){

			//delete[] Y;
			delete[] c;
			delete[] msgl;
			delete[] msgr;
			delete[] sumcol;
			delete[] sumrow;
			//delete[] grad;
			if (inside != NULL){
				delete[] inside;
				inside = NULL;
			}
			act_set.clear();
			sorted_ever_act_c.clear();
			ever_nnz_msg_l.clear();
			ever_nnz_msg_r.clear();
			delete[] is_ever_nnz_l;
			delete[] is_ever_nnz_r;
		
			if (accelerate){	
				delete last_msg_hat_l;
				delete last_msg_hat_r;
			}
		}

		void fill_act_set(){
			act_set.clear();
			for (int kk = 0; kk < K1*K2; kk++){
				act_set.push_back(make_pair(kk, 0.0));
				inside[kk] = true;
				adding_ever_act_l(kk / K2);
				adding_ever_act_r(kk % K2);
			}
		}

		inline void fill_inside(){
			for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				inside[it->first] = true;
			}
		}

		inline void clean_inside(){
			for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				inside[it->first] = false;
			}
		}

		Float gmin = 1e100;
		inline void naive_search(){
			fill_inside();
			//compute gradient

			//find max (-gradient)
			gmin = 1e100;
			int best_index = -1;
			Float g = 0.0;
			for (Int k1 = 0; k1 < K1; k1++){
				int Kk = K2*k1; 
				Float msgl_k1 = msgl[k1];
				for (Int k2 = 0; k2 < K2; k2++){
					if (inside[Kk+k2]) continue;
					g = c[Kk+k2] + rho * (msgl_k1 + msgr[k2]);
					if (g < gmin){
						gmin = g;
						best_index = Kk+k2;
					}
				}
			}
			searched_index = best_index;
			if (best_index != -1){
				act_set.push_back(make_pair(best_index, 0.0));
				//inside[max_index] = true;
			}
			//cerr << "gmax=" << gmax << ", searched_index=" << searched_index << ", ever_l="<< l->is_ever_act[searched_index / K2] << ", ever_r=" << r->is_ever_act[searched_index % K2]
			//    << ", c=" << c[searched_index] << ", msgl=" << msgl[searched_index / K2] << ", msgr=" << msgr[searched_index % K2] << endl;
			clean_inside();
		}

		inline void adding_ever_act_l(int k1){
			if (is_ever_nnz_l[k1]){
				return;
			}
			is_ever_nnz_l[k1] = true;
			ever_nnz_msg_l.push_back(k1);

			updated = true;
			//k1 is added to l->ever_act_set
			int offset = k1*K2;
			for (vector<int>::iterator it = ever_nnz_msg_r.begin(); it != ever_nnz_msg_r.end(); it++){
				//coordinate (k1, *it) should be tracked now
				int k1k2 = offset + (*it);
				sorted_ever_act_c.push_back(make_pair(c[k1k2], k1k2));
			}
		}

		inline void adding_ever_act_r(int k2){
			if (is_ever_nnz_r[k2]){
				return;
			}
			is_ever_nnz_r[k2] = true;
			ever_nnz_msg_r.push_back(k2);

			updated = true;
			//k2 is added to r->ever_act_set
			for (vector<int>::iterator it = ever_nnz_msg_l.begin(); it != ever_nnz_msg_l.end(); it++){
				//coordinate (*it, k2) should be tracked now
				int k1k2 = (*it)*K2 + k2;
				sorted_ever_act_c.push_back(make_pair(c[k1k2], k1k2));
			}
		}

		inline void update_sorted_ever_act_c(){
			//check_num = (int)trunc(size/(sqrt(size)/4+1));
			//cerr << "updating..." ;
			if (check_num < sorted_ever_act_c.size()){
				nth_element(sorted_ever_act_c.begin(), sorted_ever_act_c.begin() + check_num, sorted_ever_act_c.end());
				sorted_ever_act_c.erase(sorted_ever_act_c.begin() + check_num, sorted_ever_act_c.end());
			}
			//cerr << "done" << endl;
		}

		/*inline void fast_search(){
			vector<int> pos_left, pos_right;
			for (int k = 0; k < K1; k++){
				if (msgl[k] < 0)
					pos_left.push_back(k);
			}
			for (int k = 0; k < K2; k++){
				if (msgr[k] < 0)
					pos_right.push_back(k);
			}
			stats->area1 += pos_left.size() + pos_right.size();
		}*/

		//bi_search()
		inline void search(){
			fill_inside();
			if (K1 * K2 < 500){
				naive_search();
			}
			stats->bi_search_time -= get_current_time();
			if (updated){
				updated = false;
				update_sorted_ever_act_c();
			}
			//naive_search();
			//return;
			//fast_search();
			//find argmin_{k_1, k_2} gradient(k1, k2) = c[k1*K2+k2] + rho(msgl[k1] + msgr[k2])  
			int min_index = -1;
			Float gmin = 1e100;

			//area 4: msgl = 0, msgr = 0
			bool* ever_nnz_l = is_ever_nnz_l;
			bool* ever_nnz_r = is_ever_nnz_r;
			if (double_zero_area_index < K1*K2){
				pair<Float, int> d = sorted_c[double_zero_area_index];
				//maintain stats of search 
				//stats->area4++;
				////////
				while (double_zero_area_index < K1*K2 && (ever_nnz_l[d.second / K2] || ever_nnz_r[d.second % K2])){
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

			//stats->bi_search_time += get_current_time();
			//stats->maintain_time -= get_current_time();
			//preprocessing
			vector<pair<Float, int>> nnz_left;
			for (vector<int>::iterator it_l = ever_nnz_msg_l.begin(); it_l != ever_nnz_msg_l.end(); it_l++){
				int k = *it_l;
				Float msgl_k = msgl[k];
				if (fabs(msgl_k) > 1e-12)
					nnz_left.push_back(make_pair(msgl_k, k));
			}
			vector<pair<Float, int>> nnz_right;
			for (vector<int>::iterator it_r = ever_nnz_msg_r.begin(); it_r != ever_nnz_msg_r.end(); it_r++){
				int k = *it_r;
				Float msgr_k = msgr[k];
				if (fabs(msgr_k) > 1e-12)
					nnz_right.push_back(make_pair(msgr_k, k));
			}
			int nnz_l = nnz_left.size();
			int nnz_r = nnz_right.size();
			//maintain stats of search 
			//sorted in decreasing order
			sort(nnz_left.begin(), nnz_left.end(), less<pair<Float, int>>());
			sort(nnz_right.begin(), nnz_right.end(), less<pair<Float, int>>());
			int* record_l = new int[K1];
			for (int i = 0; i < K1; i++)
				record_l[i] = -1;

			//int num = (int)(50*nnz_l + 50*nnz_r + 2);
			//if (num > K1*K2)
			//    num = K1*K2;
			vector<pair<Float, int>>::iterator tail;
			if (check_num < sorted_ever_act_c.size()){
				tail = sorted_ever_act_c.begin() + check_num; // guaranteed to be smaller than .end()
			} else {
				tail = sorted_ever_act_c.end();
			}
			stats->area1 += (tail - sorted_ever_act_c.begin());
			//cerr << "entered " << sorted_ever_act_c.size() << ", check=" << check_num<< endl;
			for (vector<pair<Float, int>>::iterator it = sorted_ever_act_c.begin(); it != tail; it++){
				int k1k2 = it->second;
				if (inside[k1k2]) continue;
				int k1 = k1k2 / K2, k2 = k1k2 % K2;
				Float g = it->first + rho * (msgl[k1] + msgr[k2]);
				if (g < gmin){
					gmin = g;
					min_index = k1k2;
				}
				if (record_l[k1] == -1 || msgr[record_l[k1]] > msgr[k2]){
					record_l[k1] = k2;
				}
			}
			Float msgr_lower = (nnz_r==0)?0.0:nnz_right[0].first;
			Float msgl_lower = (nnz_l==0)?0.0:nnz_left[0].first;
			//Float c_lower = (check_num == 0)?sorted_c[0].first:sorted_ever_act_c[check_num-1].first;
			//stats->bi_search_time -= get_current_time();
			//stats->maintain_time += get_current_time();
			/*
			   for (int i = 0; i < nnz_l; i++)
			   cerr << index_l[i] << ":" << msgl[index_l[i]] << " ";
			   cerr << endl;
			   for (int i = 0; i < nnz_r; i++)
			   cerr << index_r[i] << ":" << msgr[index_r[i]] << " ";
			   cerr << endl;
			 */
			////////
			if (nnz_l > 0 && nnz_r > 0){
				//cerr << "gmin=" << gmin << ", min_index=" << min_index << ", ever_l="<< l->is_ever_act[min_index / K2] << ", ever_r=" << r->is_ever_act[min_index % K2]
				//    << ", c=" << c[min_index]<< ", msgl=" << msgl[min_index / K2] << ", msgr=" << msgr[min_index % K2] << endl;
				//area 1: msgl != 0, msgr != 0, search every entry
				int limit = -1;
				for (vector<pair<Float, int>>::iterator it_l = nnz_left.begin(); it_l != nnz_left.end(); it_l++){
					int k1 = it_l->second;
					int offset = k1*K2;
					Float msgl_k1 = it_l->first;
					if (record_l[k1] != -1 && (limit == -1 || msgr[limit] > msgr[record_l[k1]])){
						limit = record_l[k1];
					}
					if (msgl_k1 + sorted_row[k1][0].first + msgr_lower >= gmin)
						continue;
					//cerr << "k1=" << k1 << ", limit=" << limit << ", record[k1]=" << record_l[k1]<< endl;
					if (limit == nnz_right[0].second)
						break;
					for (vector<pair<Float, int>>::iterator it_r = nnz_right.begin(); it_r != nnz_right.end(); it_r++){
						stats->area1++;
						int k2 = it_r->second;
						if (k2 == limit)
							break;
						int k1k2 = offset + k2;
						if (inside[k1k2]) continue;
						Float g = c[k1k2] + rho*(msgl_k1 + it_r->first);
						if (g < gmin){
							gmin = g;
							min_index = offset + k2;
						}
					}
				}
			}
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

			//area 2: msgl != 0, msgr = 0, 
			//subtask: find gmin_{k1} for each row (k1, :).
			//Here msgl_k1 is fixed, c[k1, k2] + rho*msgl_k1 + rho*msgr[k2] >= c[k1, k2] + rho*msgl_k1.
			//We visit (k1, :) in increasing order of c. Notice that:
			//1. At any time (even include (k1,k2) that is inside act_set), (c[k1, k2] + rho*msgl_k1) is a lower bound of gmin_{k1}. 
			//2. If we ever found that (!inside[k1, k2] && msgr[k2] <= 0), then (c[k1, k2] + rho*msgl_k1 + rho*msgr[k2]) can dominate this row. And we can just stop here. 
			for (vector<pair<Float, int>>::iterator it_l = nnz_left.begin(); it_l != nnz_left.end(); it_l++){
				int k1 = it_l->second;
				int offset = k1*K2;
				Float msgl_k1 = it_l->first;
				pair<Float, int>* sorted_row_k1 = sorted_row[k1];
				if (msgl_k1 + sorted_row_k1[0].first + msgr_lower >= gmin)
					continue;
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
					Float msgr_k2 = msgr[k2];
					//try to update gmin
					if (gmin > val + rho * (msgl_k1 + msgr_k2) ){
						gmin = val + rho * (msgl_k1 + msgr_k2) ;
						min_index = offset + k2;
					}
					if (msgr_k2 <= 0){
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
			for (vector<pair<Float, int>>::iterator it_r = nnz_right.begin(); it_r != nnz_right.end(); it_r++){
				int k2 = it_r->second;
				Float msgr_k2 = it_r->first;
				pair<Float, int>* sorted_col_k2 = sorted_col[k2];
				if (msgr_k2 + sorted_col_k2[0].first + msgl_lower >= gmin)
					continue;
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
					Float msgl_k1 = msgl[k1];
					if (gmin > val + rho * (msgl_k1 + msgr_k2) ){
						gmin = val + rho * (msgl_k1 + msgr_k2) ;
						min_index = k1k2;
					}
					if (msgl_k1 <= 0){
						//this is a dominator, since it's already updated, we simply break here.
						break;
					}
					if (val + rho * msgr_k2 > gmin){
						//lower bound is larger than current min value, 
						break;
					}
				}
			}

			delete[] record_l;

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
			//   << ", c=" << c[min_index]<< ", msgl=" << msgl[min_index / K2] << ", msgr=" << msgr[min_index % K2] << endl;
			//assert(fabs(gmin+gmax) < 1e-6);
			searched_index = min_index;
			stats->bi_search_time += get_current_time();
			clean_inside();
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
			if (act_set.size() == 0){
				return;
			}
			Float A = rho * (2 + ever_nnz_msg_l.size() + ever_nnz_msg_r.size());
			Float* Y_new = new Float[act_set.size()];
			stats->bi_subsolve_time -= get_current_time();
			int act_count = 0;
			if (soft_bcfw){
				Float gmin = 1e300;
				Float dot_prod = 0.0;
				int min_index = -1;
				Float l1_sum = 0.0;
				for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++){
					int k1k2 = it->first;
					int k1 = k1k2/K2, k2 = k1k2 % K2;
					Float g = c[k1k2] + rho*(msgl[k1] + msgr[k2]);
					dot_prod += g * it->second;
					l1_sum += it->second;
					if (g < gmin){
						gmin = g;
						min_index = k1k2;
					}
				}
				if (l1_sum < 1.0){
					for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++){
						Y_new[it->first] = 0.0;
						if (it->first == sorted_c[0].second)
							Y_new[it->first] = 1.0;
					}
				} else {
					if (dot_prod - gmin <= 1e-12){

						stats->bi_subsolve_time += get_current_time();
						delete[] Y_new;
						return;
					}

					//up = -<grad, \Delta Y>, down = A/2 \|\Delta Y\|_2^2
					Float up = 0.0, down = 0.0;
					//Float A = 2*rho;
					for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++){
						int k1k2 = it->first;
						int k1 = k1k2/K2, k2 = k1k2 % K2;
						Float g = c[k1k2] + rho*(msgl[k1] + msgr[k2]);
						Float delta_Y = -it->second;
						if (k1k2 == min_index){
							delta_Y += 1;
						}
						up -= g * delta_Y;
						down += A * delta_Y * delta_Y;
					}
					//cerr << "down=" << down << endl;
					//assert(fabs(down)>1e-12);
					Float gamma = up/down;
					if (gamma < 0)
						gamma = 0;
					if (gamma > 1)
						gamma = 1;
					act_count = 0;
					for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
						int k1k2 = it->first;
						Float delta_Y = -it->second;
						if (k1k2 == min_index){
							delta_Y += 1;
						}
						Y_new[act_count] = it->second + delta_Y * gamma;
					}
				}
			} else {
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
					weight_b = 0.0;
					for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
						int k1k2 = it->first;
						int k1 = k1k2 / K2;
						int k2 = k1k2 % K2;
						C[act_count] = -(c[k1k2] + rho * (msgl[k1] + msgr[k2] - sumrow[k1] - sumcol[k2] ))/A + it->second; // C = -B/A
						if (C[act_count] > 0.0){
							weight_b += C[act_count];
						}
					}
					stats->weight_b += weight_b;
					solve_simplex(act_set.size(), Y_new, C);
					delete[] C;
				}
			}

			//fill_inside();
			//update Y and message
			act_count = 0;
			
			for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k1k2 = it->first;
				if (fabs(Y_new[act_count]) < nnz_tol){
					Y_new[act_count] = 0.0;
				}
				Float delta_Y = Y_new[act_count] - it->second;
				stats->delta_Y_l1 += fabs(delta_Y);
				int k1 = k1k2 / K2, k2 = k1k2 % K2;
				msgl[k1] += delta_Y; sumrow[k1] += delta_Y;
				msgr[k2] += delta_Y; sumcol[k2] += delta_Y;
			}

			vector<pair<int, Float>> next_act_set;
			next_act_set.clear();
			act_count = 0;
			bool update = false;
			for(vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k1k2 = it->first;
				it->second = Y_new[act_count];
				//possible shrink here
				if (!shrink || it->second > nnz_tol){
					int k1 = k1k2 / K2, k2 = k1k2 % K2;
					if (k1k2 == searched_index){
						adding_ever_act_l(k1);
						adding_ever_act_r(k2);
					}
					next_act_set.push_back(make_pair(k1k2, it->second));
				} else {
					//inside[k1k2] = false;
				}
			}

			act_set = next_act_set;

			delete[] Y_new;
			stats->bi_subsolve_time += get_current_time();
			//clean_inside();
		}

		//     \mu^{t+1} - \mu^t
		//   = \msg^{t+1} - \msg^t
		//   = eta * ( (\sum_j y[:][j]) - y[:])
		Float lambda_t = 0.0;
		inline void update_multipliers(){
			stats->maintain_time -= get_current_time();
			//update msgl, msgr
			Float* y_l = l->y;
			Float* y_r = r->y;
			if (accelerate){
				Float lambda_tplus = (1 + sqrt(1 + 4*lambda_t*lambda_t))/2.0;
				Float gamma = (1 - lambda_t)/lambda_tplus;
					
				for (vector<int>::iterator it_l = ever_nnz_msg_l.begin(); it_l != ever_nnz_msg_l.end(); it_l++){
					int k = *it_l;
					msgl[k] += eta * (sumrow[k] - y_l[k])*(1.0-gamma) + gamma * (last_msg_hat_l[k]);
					last_msg_hat_l[k] = (msgl[k] - last_msg_hat_l[k]*gamma)/(1.0-gamma);
				}
				for (vector<int>::iterator it_r = ever_nnz_msg_r.begin(); it_r != ever_nnz_msg_r.end(); it_r++){
					int k = *it_r;
					msgr[k] += eta * (sumcol[k] - y_r[k])*(1.0-gamma) + gamma * (last_msg_hat_r[k]);
					last_msg_hat_r[k] = (msgr[k] - last_msg_hat_r[k]*gamma)/(1.0-gamma);
				}
				
				lambda_t = lambda_tplus;
			} else {
				for (vector<int>::iterator it_l = ever_nnz_msg_l.begin(); it_l != ever_nnz_msg_l.end(); it_l++){
					int k = *it_l;
					msgl[k] += eta * (sumrow[k] - y_l[k]);
				}
				for (vector<int>::iterator it_r = ever_nnz_msg_r.begin(); it_r != ever_nnz_msg_r.end(); it_r++){
					int k = *it_r;
					msgr[k] += eta * (sumcol[k] - y_r[k]);
				}
			}
			//msgr[best_k] += etar * (sumcol[best_k] - y_r[best_k]);

			stats->maintain_time += get_current_time();
		}

		inline int nnz_msg(){
			int nnz1 = 0;
			for (int i = 0; i < K1; i++){
				if (msgl[i] < -nnz_tol)
					nnz1++;
			}
			int nnz2 = 0;
			for (int i = 0; i < K2; i++){
				if (msgr[i] < -nnz_tol)
					nnz2++;
			}
			return nnz1*nnz2;
		}

		inline Float score(){
			/*Float score = 0.0;
			for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k1k2 = it->first;
				score += c[k1k2]*it->second;
			}
			return score;
			*/
			int k1k2 = l->recent_pred * K2 + r->recent_pred;
			//cerr << "recent_pred=(" << l->recent_pred << "," << r->recent_pred << "), c=" << c[k1k2] << endl;
			return c[k1k2];
		}

        inline Float rel_score(){
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
				/*if (inf > p_inf)
					p_inf = inf;*/
				p_inf += inf;
			}
			for (int k = 0; k < K2; k++){
				Float inf = fabs(sumcol[k] - y_r[k]);
				/*if (inf > p_inf)
					p_inf = inf;*/
				p_inf += inf;
			}
			return p_inf;
		}

		inline void display(){
			cerr << endl;
			cerr << "msgl:\t";
			for (int i = 0; i < K1; i++){
				cerr << msgl[i] << " ";
			}
			cerr << endl;
			cerr << "msgr:\t";
			for (int i = 0; i < K2; i++){
				cerr << msgr[i] << " ";
			}
			cerr << endl;
			for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int kk = it->first;
				cerr << "(" << (kk/K2) << "," << (kk%K2) << ")" << ":" << it->second << ":" << c[kk] << " ";
			}
			cerr << endl;
		}

		int dinf_index = -1;
		//bi_dual_inf()
		inline Float dual_inf(){
			Float gmin = 1e300;
			dinf_index = -1;
			Float dot_prod = 0.0;
			int min_index = -1;
			for (vector<pair<int, Float>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k1k2 = it->first;
				int k1 = k1k2/K2, k2 = k1k2 % K2;
				Float g = c[k1k2] + rho*(msgl[k1] + msgr[k2]);
				dot_prod += g * it->second;
				if (g < gmin){
					gmin = g;
					min_index = k1k2;
				}
			}
			/*if (dot_prod > gmin + 1e-6){
				cerr << "dot_prod=" << dot_prod << ", gmin=" << gmin << ", min_index=" << min_index << endl;
			}*/
			if (dot_prod - gmin > 0.0){
				dinf_index = min_index;
			}
			return dot_prod - gmin;
		}
};

inline void UniFactor::adding_ever_act(int k){
	/*if (is_ever_act[k]){
	  cerr << "NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO" << endl;
	  return;
	  }*/
	//is_ever_act[k] = true;
	//ever_act_set.push_back(k);
	for (vector<BiFactor*>::iterator to_r = edge_to_right.begin(); to_r != edge_to_right.end(); to_r++){
		BiFactor* edge = *to_r;
		edge->adding_ever_act_l(k);
	}
	for (vector<BiFactor*>::iterator to_l = edge_to_left.begin(); to_l != edge_to_left.end(); to_l++){
		BiFactor* edge = *to_l;
		edge->adding_ever_act_r(k);
	}
}

bool* BiFactor::inside = new bool[4000000];

#endif
