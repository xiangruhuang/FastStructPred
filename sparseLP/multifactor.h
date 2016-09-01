#ifndef MULTIFACTOR_H
#define MULTIFACTOR_H

#include "factor.h"
#include "util.h"

extern Stats* stats;

extern bool debug;
bool shrink = true;

class MultiBiFactor;

//Multilabel version of UniFactor, (i.e. without simplex constraint)
class MultiUniFactor : public Factor{
	public:
		//fixed
		int K;
		Float rho, eta;
		Float* c;
		Float nnz_tol;

		//maintained internally
		vector<pair<Float, int>> act_set; // <nnz_y, active_index>
		bool* inside;
		bool* is_ever_nnz;
		int searched_index;
		int* deg;
		Float* grad;
		Float* cache;
		//only uses edge->sum_Y_bar

		//will be accessed && should be maintained
		vector<pair<Float, int>> ever_nnz_y_bar; // <y_bar, index>
		Float* y;
		Float* y_bar;
		MultiBiFactor* edge; // assume only one edge connects to it

		int* recent_pred;
		inline MultiUniFactor(int _K, Float* _c, Param* param){
			K = _K;
			rho = param->rho;
			eta = param->eta;
			nnz_tol = param->nnz_tol;
			c = _c;

			//edge will be assigned later

			//cache
			grad = new Float[K];
			memset(grad, 0, sizeof(Float)*K);
			cache = new Float[K];
			memset(cache, 0, sizeof(Float)*K);

			deg = new int[K];
			memset(deg, 0, sizeof(int)*K);

			inside = new bool[K];
			memset(inside, 0, sizeof(bool)*K);
			is_ever_nnz = new bool[K];
			memset(is_ever_nnz, 0, sizeof(bool)*K);
			y = new Float[K];
			memset(y, 0, sizeof(Float)*K);
			y_bar = new Float[K];
			memset(y_bar, 0, sizeof(Float)*K);

			//fill_act_set();
			recent_pred = new int[K];
		}

		~MultiUniFactor(){
			delete[] inside;
			delete[] is_ever_nnz;
			delete[] y;
			delete[] y_bar;
			delete[] grad;
			delete[] cache;
			delete[] deg;
		}

		inline void fill_act_set(){
			for (int i = 0; i < K; i++){
				act_set.push_back(make_pair(0.0, i));
				inside[i] = true;
				is_ever_nnz[i] = true;
				ever_nnz_y_bar.push_back(make_pair(0.0, i));
			}
		}

		inline void search();

		//min_y <c_{k1}, y(k1)> + rho/2 \sum_{k2} (\| y(k1) - [Y^{10}_bar(k1, k2) + Y^{11}_bar(k1, k2) - (y_bar(k1) - y(k1))] \|_2^2
		//             + \| y(k1) - [Y^{11}_bar(k2, k1) + Y^{01}_bar(k2, k1) - (y_bar(k1) - y(k1))] \|_2^2 )
		// y(k1)^+ = c_{k1} + rho \sum_{k2} 
		//              (y(k1) - [Y^{10}_bar(k1, k2) + Y^{11}_bar(k1, k2) - (y_bar(k1) - y(k1))]
		//               + y(k1) - [Y^{11}_bar(k2, k1) + Y^{01}_bar(k2, k1) - (y_bar(k1) - y(k1))]  )

		inline void subsolve();

		inline void update_multipliers(){
			stats->maintain_time -= get_current_time();
			for (vector<pair<Float, int>>::iterator it = ever_nnz_y_bar.begin(); it != ever_nnz_y_bar.end(); it++){
				int k = it->second;
				it->first += y[k] * eta;
				y_bar[k] = it->first;
			}
			stats->maintain_time += get_current_time();
		}

		inline Float score(){
			Float score = 0.0;
			memset(recent_pred, 0, sizeof(int)*K);
			for (vector<pair<Float, int>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = it->second;
				if (it->first > 0.5){
					recent_pred[k] = 1;
					score += c[k];
				}
			}
			return score;
		}

		inline void display(){

			cerr << endl;
			cerr << "y:\t";
			for (vector<pair<Float, int>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = it->second;
				cerr << k << ":" << y[k] << " ";
			}
			cerr << endl;
			cerr << "ybar:\t";
			for (vector<pair<Float, int>>::iterator it = ever_nnz_y_bar.begin(); it != ever_nnz_y_bar.end(); it++){
				int k = it->second;
				cerr << k << ":" << y_bar[k] << " ";
			}

			cerr << endl;
			/*for (int k = 0; k < K; k++)
			  cerr << y[k] << " ";
			  cerr << endl;*/
		}

		int dinf_index = -1;
		inline Float dual_inf();


		//y_bar == ever_nnz_y_bar
		//y == act_set
		inline void check_integrity(){
			for (vector<pair<Float, int>>::iterator it = ever_nnz_y_bar.begin(); it != ever_nnz_y_bar.end(); it++){
				assert(fabs(it->first - y_bar[it->second]) < nnz_tol);
			}
			for (vector<pair<Float, int>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				assert(fabs(it->first - y[it->second]) < nnz_tol);
			}
		}

};

class MultiBiFactor : public Factor{
	public:

		//fixed
		int K;
		Float rho, eta;
		Float* c;
		pair<Float, int>* sorted_c;
		Float nnz_tol;

		Float** Y;
		Float** Y_bar;

		//maintained
		vector<pair<Float*, int>> act_set; // <active_y, active_index>
		bool* inside;
		bool* is_ever_nnz;
		int searched_index;
		MultiUniFactor* node; // assume only one node connects to it
		int search_pointer;

		//can be accessed and should be maintained
		vector<pair<Float*, int>> ever_nnz_Y_bar; // <y_bar, index>
		Float* sum_Y_bar; // sum_Y_bar(k) = \sum_{k2} (Y_bar^01(k2, k) + Y_bar^{11}(k2, k) + Y_bar^10(k, k2) + Y_bar^{11}(k, k2))


		MultiBiFactor(int _K, MultiUniFactor* _node, ScoreVec* sv, Param* param){
			K = _K;
			c = sv->c;
			sorted_c = sv->sorted_c;
			rho = param->rho;
			eta = param->eta;
			nnz_tol = param->nnz_tol;
			node = _node;
			node->edge = ((MultiBiFactor*)this);

			Y = new Float*[K*K];
			for (int kk = 0; kk < K*K; kk++){
				Y[kk] = zero_cube();
			}
			Y_bar = new Float*[K*K];
			for (int kk = 0; kk < K*K; kk++){
				Y_bar[kk] = zero_cube();
			}
			sum_Y_bar = new Float[K];
			memset(sum_Y_bar, 0.0, sizeof(Float)*K);
			inside = new bool[K*K];
			memset(inside, 0, sizeof(bool)*K*K);
			for (int k1 = 0; k1 < K; k1++){
				for (int k2 = 0; k2 <= k1; k2++){
					inside[k1*K+k2] = true;
				}
			}
			is_ever_nnz = new bool[K*K];
			memset(is_ever_nnz, 0, sizeof(bool)*K*K);
			search_pointer = 0;

			//just for test
			//fill_act_set();
		}

		~MultiBiFactor(){
			for (int k = 0; k < K*K; k++){
				if (Y[k] != NULL)
					delete[] Y[k];
			}
			delete[] Y;
			for (int k = 0; k < K*K; k++){
				if (Y_bar[k] != NULL)
					delete[] Y_bar[k];
			}
			delete[] Y_bar;
			delete[] sum_Y_bar;
			delete[] is_ever_nnz;
			delete[] inside;
		}

		inline Float* zero_cube(){
			Float* cube = new Float[4];
			cube[0] = 0;
			cube[1] = 0;
			cube[2] = 0;
			cube[3] = 0;
			return cube;
		}

		inline void fill_act_set(){
			for (int k1 = 0; k1 < K; k1++){
				for (int k2 = k1+1; k2 < K; k2++){
					int kk = k1*K+k2;
					act_set.push_back(make_pair(Y[kk], kk));
					inside[kk] = true;
					ever_nnz_Y_bar.push_back(make_pair(Y_bar[kk], kk));
					is_ever_nnz[kk] = true;
					node->deg[k1]++;
					node->deg[k2]++;
				}
			}
		}

		inline void naive_search(){
			Float* y_bar = node->y_bar;
			Float gmin = 0.0;
			int min_index = -1;

			int k1k2 = sorted_c[search_pointer].second;
			while (search_pointer < K*K-1 && inside[k1k2]){
				search_pointer++;
				k1k2 = sorted_c[search_pointer].second;
			}
			if (sorted_c[search_pointer].first < 0.0){
				min_index = k1k2;
			}

			/*for (int k1 = 0; k1 < K; k1++){
			  for (int k2 = k1 + 1; k2 < K; k2++){
			  int k1k2 = k1*K+k2;
			  assert(k1 < k2);
			  if (inside[k1k2]) continue;
			  Float* Ybar = Y_bar[k1k2];
			  assert(fabs(Ybar[3]) < nnz_tol);
			  assert(fabs(Ybar[2]) < nnz_tol);
			  assert(fabs(Ybar[1]) < nnz_tol);
			  assert(fabs(y_bar[k1]) < nnz_tol || fabs(y_bar[k2]) < nnz_tol);
			  Float msgl = Ybar[3] + Ybar[2] - y_bar[k1];
			  Float msgr = Ybar[3] + Ybar[1] - y_bar[k2];
			  Float g = c[k1k2]; //+ rho * (msgl + msgr);
			  if (g < gmin){
			  gmin = g;
			  min_index = k1k2;
			  }
			  }
			  }*/
			searched_index = min_index;
			if (debug){
				//cerr << "bi_searched_index=" << searched_index << ", grad=" << gmin << ", c*=" << c[searched_index] << endl;
			}
			if (min_index != -1){
				inside[min_index] = true;
				int k1 = min_index / K, k2 = min_index % K;
				Float yl = node->y[k1];
				Float yr = node->y[k2];
				Float* Ykk = Y[min_index];
				Float* Ybar = Y_bar[min_index];
				Ykk[3] = 0;
				Ykk[2] = yl;
				Ykk[1] = yr;
				Ykk[0] = 1-yr-yl;
				Ybar[3] = 0;
				Ybar[2] = y_bar[k1];
				Ybar[1] = y_bar[k2];
				Ybar[0] = 1-yl-yr;
				sum_Y_bar[k1] += y_bar[k1];
				sum_Y_bar[k2] += y_bar[k2];
				act_set.push_back(make_pair(Ykk, min_index));
			}
		}

		//bi_search()
		inline void search(){
			stats->bi_search_time -= get_current_time();
			naive_search();
			stats->bi_search_time += get_current_time();
		}

		//bi_subsolve()
		inline void subsolve(){
			if (act_set.size() == 0)
				return;
			stats->bi_subsolve_time -= get_current_time();        

			Float A = rho * 4;
			Float* Y_new = new Float[4]; 

			int act_count = 0;
			vector<pair<Float*, int>> next_act_set;
			if (fabs(rho) < 1e-12){
				act_count = 0;
				for (vector<pair<Float*, int>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
					int k1k2 = it->second;
					Float* Y_old = it->first;
					if (c[k1k2] <= 0){
						Y_new[3] = 1.0;
						Y_new[2] = 0.0;
						Y_new[1] = 0.0;
						Y_new[0] = 0.0;
					} else {
						Y_new[3] = 0.0;
						Y_new[2] = 0.0;
						Y_new[1] = 0.0;
						Y_new[0] = 1.0;
					}
					int k1 = k1k2 / K, k2 = k1k2 % K;
					Float* Ybar = Y_bar[k1k2];
					//01
					Float delta_y = Y_new[1] - Y_old[1];
					sum_Y_bar[k2] += delta_y;
					Ybar[1] += delta_y;
					//10
					delta_y = Y_new[2] - Y_old[2];
					sum_Y_bar[k1] += delta_y;
					Ybar[2] += delta_y;
					//11
					delta_y = Y_new[3] - Y_old[3];
					sum_Y_bar[k2] += delta_y;
					sum_Y_bar[k1] += delta_y;
					Ybar[3] += delta_y;

					//Y[k1k2] = new_cube;

					Y_old[3] = Y_new[3];
					Y_old[2] = Y_new[2];
					Y_old[1] = Y_new[1];
					Y_old[0] = Y_new[0];
					//shrink
					if (true || fabs(Y_new[3]) > nnz_tol ){
						next_act_set.push_back(make_pair(Y_old, k1k2));
						if (!is_ever_nnz[k1k2]){
							is_ever_nnz[k1k2] = true;
							//cerr << searched_index / K << " " << searched_index % K << endl;
							ever_nnz_Y_bar.push_back(make_pair(Ybar, searched_index));
						}
					} else {
						inside[k1k2] = false;
					}

				}

			} else {
				Float* b = new Float[4];
				act_count = 0;
				Float* y_bar = node->y_bar;
				for (vector<pair<Float*, int>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
					int k1k2 = it->second;
					int k1 = k1k2 / K, k2 = k1k2 % K;
					Float* Y_old = it->first;
					Float* Ybar = Y_bar[k1k2];
					//Float ybar_k1 = y_bar[k1], ybar_k2 = y_bar[k2];
					Float ml = rho/A*(Ybar[2] + Ybar[3] - y_bar[k1]);
					Float mr = rho/A*(Ybar[1] + Ybar[3] - y_bar[k2]);

					if (debug && k1k2 == dinf_index){
						//cerr << "k1k2=" << k1k2 << ", ml=" << ml << ", mr=" << mr << endl;
					}
					b[0] = Y_old[0];
					b[1] = Y_old[1] - mr;
					b[2] = Y_old[2] - ml;
					b[3] = Y_old[3] - (ml + mr) - (c[k1k2] / A);

					//if (k1k2 == 74)
					//    cerr << "k1k2=" << k1k2 << ", c=" << c[k1k2] << ", b:\t" << b[0] << " " << b[1] << " " << b[2] << " " << b[3] << ", ml=" << ml << ", mr=" << mr << endl;

					solve_simplex(4, Y_new, b);

					//update Y_bar, sum_Y_bar, Y

					//01
					Float delta_y = Y_new[1] - Y_old[1];
					sum_Y_bar[k2] += delta_y;
					Ybar[1] += delta_y;
					Y_old[1] = Y_new[1];
					//10
					delta_y = Y_new[2] - Y_old[2];
					sum_Y_bar[k1] += delta_y;
					Ybar[2] += delta_y;
					Y_old[2] = Y_new[2];
					//11
					delta_y = Y_new[3] - Y_old[3];
					sum_Y_bar[k2] += delta_y;
					sum_Y_bar[k1] += delta_y;
					Ybar[3] += delta_y;
					Y_old[3] = Y_new[3];

					//Y[k1k2] = new_cube;

					Y_old[0] = Y_new[0];
					//shrink
					if (true || fabs(Y_new[3]) > nnz_tol ){
						//next_act_set.push_back(make_pair(Y_old, k1k2));
						if (!is_ever_nnz[k1k2]){
							is_ever_nnz[k1k2] = true;
							ever_nnz_Y_bar.push_back(make_pair(Ybar, searched_index));
							//cerr << searched_index / K << " " << searched_index % K << endl;
							node->deg[k1]++; node->deg[k2]++;
						}
					} else {
						inside[k1k2] = false;
					}

				}
			}

			stats->bi_subsolve_time += get_current_time();            

			//act_set = next_act_set;

			//update ever_nnz_Y_bar
			/*for (vector<pair<Float*, int>>::iterator it = ever_nnz_Y_bar.begin(); it != ever_nnz_Y_bar.end(); it++){
			  int k1k2 = it->second;
			  Float* Ybar = it->first;
			  }*/

			delete[] Y_new;

		}

		//update Y_bar, sum_Y_bar
		inline void update_multipliers(){
			stats->maintain_time -= get_current_time();
			for (vector<pair<Float*, int>>::iterator it = ever_nnz_Y_bar.begin(); it != ever_nnz_Y_bar.end(); it++){
				int k1k2 = it->second;
				Float* YY = Y[k1k2];
				Float* Ybar = it->first;
				Ybar[2] += eta*YY[2];
				Ybar[1] += eta*YY[1];
				Ybar[3] += eta*YY[3];
				int k1 = k1k2 / K, k2 = k1k2 % K;
				sum_Y_bar[k1] += eta*(YY[2] + YY[3]);
				sum_Y_bar[k2] += eta*(YY[1] + YY[3]);
			}
			stats->maintain_time += get_current_time();
		}

		inline Float func_val(int k1k2){
			Float* ybar = node->y_bar;
			Float* Ybar = Y_bar[k1k2];
			Float val = 0.0;
			int k1 = k1k2 / K, k2 = k1k2 % K;
			Float msgl = Ybar[3] + Ybar[2] - ybar[k1];
			Float msgr = Ybar[3] + Ybar[1] - ybar[k2];
			val = rho/2*(msgl * msgl + msgr * msgr) + c[k1k2]*Y[k1k2][3];
			return val;
		}

		inline void display(){

			cerr << endl;
			cerr << "Y:\t";
			for (vector<pair<Float*, int>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int kk = it->second;
				int k1 = kk / K, k2 = kk % K;
				cerr << "(" << k1 << "," << k2 << ")" << ":" << it->first[0] << "," << it->first[1] << "," << it->first[2] << "," << it->first[3] << " ";
			}
			cerr << endl;
			cerr << "Ybar:\t";
			for (vector<pair<Float*, int>>::iterator it = ever_nnz_Y_bar.begin(); it != ever_nnz_Y_bar.end(); it++){
				int kk = it->second;
				int k1 = kk / K, k2 = kk % K;
				cerr << "(" << k1 << "," << k2 << ")" << ":" << it->first[0] << "," << it->first[1] << "," << it->first[2] << "," << it->first[3] << " ";
			}
			cerr << endl;

			cerr << "sum_Y_bar:\t";
			for (int k = 0; k < K; k++){
				if (fabs(sum_Y_bar[k]) > nnz_tol){
					cerr << k << ":" << sum_Y_bar[k] << " ";
				}
			}
			cerr << endl;
		}

		inline Float score(){
			Float score = 0.0;
			//for (vector<pair<Float*, int>>::iterator it = act_set.begin(); it != act_set.end(); it++){
			for (int kk = 0; kk < K*K; kk++){
				//score += it->first[3] * c[kk];
				int k1 = kk / K, k2 = kk % K;
				score += node->recent_pred[k1]*node->recent_pred[k2]*c[kk];
			}
			return score;
		}

		inline Float infea(){
			Float p_inf = 0.0;
			Float* y = node->y;
			for (vector<pair<Float*, int>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				Float* Ykk = it->first;
				int k1k2 = it->second;
				int k1 = k1k2 / K, k2 = k1k2 % K;
				//cerr << Ykk[0] << " " << Ykk[1] << " " << Ykk[2] << " " << Ykk[3] << endl;
				Float inf = fabs(Ykk[2] + Ykk[3] - y[k1]);
				if (inf > p_inf)
					p_inf = inf;
				inf = fabs(Ykk[1] + Ykk[3] - y[k2]);
				if (inf > p_inf)
					p_inf = inf;
			}
			return p_inf;
		}

		int dinf_index = -1;
		//bi_dual_inf
		inline Float dual_inf(){
			Float dinf = 0.0;
			Float g = 0.0;
			dinf_index = -1;
			Float* y_bar = node->y_bar;
			for (vector<pair<Float*, int>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				//compute gradient of each entry
				int k1k2 = it->second;
				int k1 = k1k2 / K, k2 = k1k2 % K;
				Float* Ykk = it->first;
				Float* Ybar = Y_bar[k1k2];
				Float msgl = Ybar[2] + Ybar[3] - y_bar[k1];
				Float msgr = Ybar[1] + Ybar[3] - y_bar[k2];
				Float* grad = new Float[4];
				grad[3] = c[k1k2] + rho*(msgl + msgr);
				grad[2] = rho*(msgl);
				grad[1] = rho*(msgr);
				grad[0] = 0;
				Float gmin = min(grad[1], grad[0]);
				gmin = min(grad[2], gmin);
				gmin = min(grad[3], gmin);
				Float dp = Ykk[3] * grad[3] + Ykk[2] * grad[2] + Ykk[1] * grad[1];

				if (-(gmin-dp) > dinf){
					dinf = -(gmin-dp);
					dinf_index = k1k2;
				}
				//cerr << ", gmax=" << gmax << ", dinf_index=" << dinf_index << endl;
			}
			Float* YY;
			if (dinf_index != -1){
				YY = Y[dinf_index];
			}
			if (debug){
				//cerr << "edge_dinf: dinf_index=" << dinf_index << " ,dinf=" << dinf << ", cube=(" << YY.Y00 << "," << YY.Y01 << "," << YY.Y10 << "," << YY.Y11 << ")" << endl;
			}
			return dinf;
		}

		//y_bar == ever_nnz_y_bar
		//y == act_set
		//sum_Y_bar
		inline void check_integrity(){

			for (vector<pair<Float*, int>>::iterator it = ever_nnz_Y_bar.begin(); it != ever_nnz_Y_bar.end(); it++){
				assert(it->first == Y_bar[it->second]);
			}
			for (vector<pair<Float*, int>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				assert(it->first == Y[it->second]);
			}

			//check sum_Y_bar
			Float* temp = new Float[K];
			memset(temp, 0.0, sizeof(Float)*K);
			for (vector<pair<Float*, int>>::iterator it = ever_nnz_Y_bar.begin(); it != ever_nnz_Y_bar.end(); it++){
				Float* Ybar = it->first;
				int k1k2 = it->second;
				int k1 = k1k2 / K, k2 = k1k2 % K;
				temp[k1] += Ybar[2] + Ybar[3];
				temp[k2] += Ybar[1] + Ybar[3];
			}
			for (int k = 0; k < K; k++){
				if (fabs(temp[k] - sum_Y_bar[k]) >= nnz_tol){
					cerr << "k=" << k << ", sum=" << sum_Y_bar[k] << ", temp=" << temp[k] << endl;
				}
				assert(fabs(temp[k] - sum_Y_bar[k]) < nnz_tol);
			}
			delete[] temp;
		}
};

inline void MultiUniFactor::search(){
	stats->uni_search_time -= get_current_time();
	//compute gradient = c - rho \sum msg = c + K rho y_bar - rho \sum Y_bar
	for (int k = 0; k < K; k++){
		grad[k] = c[k];
		//- rho * edge->sum_Y_bar[k];
	}
	/*for (vector<pair<Float, int>>::iterator uni_msg = ever_nnz_y_bar.begin(); uni_msg != ever_nnz_y_bar.end(); uni_msg++){
	  Float ybar = uni_msg->first;
	  int k = uni_msg->second;
	  grad[k] += rho * ybar * K;
	  }*/

	vector<pair<Float*, int>>& ever_nnz_Y_bar = edge->ever_nnz_Y_bar;
	for (vector<pair<Float*, int>>::iterator it = ever_nnz_Y_bar.begin(); it != ever_nnz_Y_bar.end(); it++){
		Float* Ybar = it->first;
		int k1k2 = it->second;
		int k1 = k1k2 / K, k2 = k1k2 % K;
		//cerr << k1 << " " << k2 << endl;
		assert(k1 < k2);
		grad[k1] -= rho * (Ybar[2] + Ybar[3] - y_bar[k1]);
		grad[k2] -= rho * (Ybar[1] + Ybar[3] - y_bar[k2]);
	}

	Float gmax = 0.0;
	int max_index = -1;
	for (int k = 0; k < K; k++){
		if (inside[k]) continue;
		//if not inside, y_k is guaranteed to be zero, and y_k is nonnegative, so we only care gradient < 0
		if (grad[k] > 0) continue;
		if (-grad[k] > gmax){
			gmax = -grad[k];
			max_index = k;
		}
	}

	searched_index = max_index;
	//cerr << "uni_searched_index=" << searched_index << ", grad=" << -gmax << endl;
	if (max_index != -1){
		act_set.push_back(make_pair(0.0, max_index));
		inside[max_index] = true;
	}
	stats->uni_search_time += get_current_time();
}

//uni_subsolve()
inline void MultiUniFactor::subsolve(){
	if (act_set.size() == 0)
		return;
	stats->uni_subsolve_time -= get_current_time();
	Float* y_new = new Float[act_set.size()];
	int act_count = 0;
	if (edge == NULL || fabs(rho) < 1e-12){
		//min_y <c, y>, no bigram at all.
		if (debug){
			//cerr << "degenerated uni_subsolve!!!!!" << endl;
		}
		act_count = 0;
		memset(y_new, 0.0, sizeof(Float)*act_set.size());
		for (vector<pair<Float, int>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
			int k = it->second;
			if (c[k] <= 0){
				y_new[act_count] = 1.0;
			}
		}
	} else {
		act_count = 0;
		Float* sum_Y_bar = edge->sum_Y_bar;
		for (vector<pair<Float, int>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
			int k = it->second;
			Float t = sum_Y_bar[k] - c[k]/rho;
			t /= deg[k];
			t -= (y_bar[k] - it->first);
			if (debug){
				//cerr << "k=" << k << ", t=" << t << endl;
			}
			//t += y[k];
			if (t > 1.0)
				t = 1.0;
			if (t < 0.0)
				t = 0.0;
			y_new[act_count] = t;
			//cerr << "k=" << k << ", t=" << t << ", sum_Y_bar=" << sum_Y_bar[k] << ", y_bar[k]=" << y_bar[k] << ", y[k]=" << y[k] << endl;
		}
	}

	//cerr << endl;
	//update y_bar, y


	vector<pair<Float, int>> next_act_set;
	next_act_set.clear();
	act_count = 0;
	for (vector<pair<Float, int>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
		int k = it->second;
		Float delta_y = y_new[act_count] - it->first;
		it->first = y_new[act_count];
		y[k] = it->first;
		//shrink
		if (!shrink || it->first >= nnz_tol){
			//this index is justed added by search()
			//Only if this index is active after subsolve, then it's added to ever_nnz_y_bar
			y_bar[k] += delta_y;
			if (k == searched_index && !is_ever_nnz[k]){
				//update act_set and ever_nnz_Y_bar of edge
				bool* ever_nnz_Y = edge->is_ever_nnz;
				vector<pair<Float*, int>>& act_set_Y = edge->act_set;
				bool* inside_Y = edge->inside;
				Float** Y = edge->Y;
				Float** Y_bar = edge->Y_bar;
				vector<pair<Float*, int>>& ever_nnz_Y_bar = edge->ever_nnz_Y_bar;

				for (vector<pair<Float, int>>::iterator itt = ever_nnz_y_bar.begin(); itt != ever_nnz_y_bar.end(); itt++){
					int k1 = min(k, itt->second);
					int k2 = max(k, itt->second);
					assert(k1 < k2);
					int k1k2 = k1*K+k2;
					if (!inside_Y[k1k2]){
						act_set_Y.push_back(make_pair(Y[k1k2],k1k2));
						inside_Y[k1k2] = true;
						ever_nnz_Y[k1k2] = true;
						ever_nnz_Y_bar.push_back(make_pair(Y_bar[k1k2], k1k2));
						deg[k1]++;
						deg[k2]++;
					}
				}

				ever_nnz_y_bar.push_back(make_pair(it->first, k));
				is_ever_nnz[k] = true;
			}
			next_act_set.push_back(make_pair(it->first, k));
		} else {
			inside[k] = false;
		}
	}
	act_set = next_act_set;

	act_count = 0;
	for (vector<pair<Float, int>>::iterator it = ever_nnz_y_bar.begin(); it != ever_nnz_y_bar.end(); it++, act_count++){
		int k = it->second;
		it->first = y_bar[k];
	}
	//cerr << endl;

	delete[] y_new;
	stats->uni_subsolve_time += get_current_time();
}

//uni_dual_inf
inline Float MultiUniFactor::dual_inf(){
	for (int k = 0; k < K; k++){
		grad[k] = c[k] - rho * edge->sum_Y_bar[k];
	}
	for (vector<pair<Float, int>>::iterator uni_msg = ever_nnz_y_bar.begin(); uni_msg != ever_nnz_y_bar.end(); uni_msg++){
		Float ybar = uni_msg->first;
		int k = uni_msg->second;
		grad[k] += rho * ybar * deg[k];
	}
	//max gradient inside active set
	Float gmax = 0.0;
	dinf_index = -1;
	for (vector<pair<Float, int>>::iterator it = act_set.begin(); it != act_set.end(); it++){
		int k = it->second;
		if (-grad[k] > gmax && (it->first < 1 - nnz_tol) ){
			gmax = -grad[k];
			dinf_index = k;
		}
		if (grad[k] > gmax && (it->first > nnz_tol) ){
			gmax = grad[k];
			dinf_index = k;
		}
	}
	if (debug){
		//cerr << "dinf_index=" << dinf_index << " ,grad=" << grad[dinf_index] << endl;
	}
	return gmax;
}
#endif
