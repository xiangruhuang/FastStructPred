#ifndef MULTIFACTOR_H
#define MULTIFACTOR_H

#include "factor.h"
#include "util.h"

extern Stats* stats;

struct Cube22{
    Float Y00, Y01, Y10, Y11;
};

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
        Float* grad;
        Float* cache;
        //only uses edge->sum_Y_bar

        //will be accessed && should be maintained
        vector<pair<Float, int>> ever_nnz_y_bar; // <y_bar, index>
        Float* y;
        Float* y_bar;
        MultiBiFactor* edge; // assume only one edge connects to it
        

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

            inside = new bool[K];
            memset(inside, 0, sizeof(bool)*K);
            is_ever_nnz = new bool[K];
            memset(is_ever_nnz, 0, sizeof(bool)*K);
            y = new Float[K];
            memset(y, 0, sizeof(Float)*K);
            y_bar = new Float[K];
            memset(y, 0, sizeof(Float)*K);
        }

        ~MultiUniFactor(){
            delete[] inside;
            delete[] is_ever_nnz;
            delete[] y;
            delete[] y_bar;
            delete[] grad;
            delete[] cache;
        }

        inline void search();

        //min_y <c_{k1}, y(k1)> + rho/2 \sum_{k2} (\| y(k1) - [Y^{10}_bar(k1, k2) + Y^{11}_bar(k1, k2) - (y_bar(k1) - y(k1))] \|_2^2
        //             + \| y(k1) - [Y^{11}_bar(k2, k1) + Y^{01}_bar(k2, k1) - (y_bar(k1) - y(k1))] \|_2^2 )
        // y(k1)^+ = c_{k1} + rho \sum_{k2} 
        //              (y(k1) - [Y^{10}_bar(k1, k2) + Y^{11}_bar(k1, k2) - (y_bar(k1) - y(k1))]
        //               + y(k1) - [Y^{11}_bar(k2, k1) + Y^{01}_bar(k2, k1) - (y_bar(k1) - y(k1))]  )
 
        inline void subsolve();

        inline void update_multipliers(){
            for (vector<pair<Float, int>>::iterator it = ever_nnz_y_bar.begin(); it != ever_nnz_y_bar.end(); it++){
                int k = it->second;
                it->first += y[k] * eta;
                y_bar[k] += y[k] * eta;
            }
        }

        inline Float Score(){
			Float score = 0.0;
			for (vector<pair<Float, int>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = it->second;
				score += c[k]*it->first;
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
};

class MultiBiFactor : public Factor{
    public:
        
        //fixed
        int K;
        Float rho, eta;
        Float* c;
        Float nnz_tol;

        Cube22* Y;
        Cube22* Y_bar;

        //maintained
        vector<pair<Cube22, int>> act_set; // <active_y, active_index>
        bool* inside;
        bool* is_ever_nnz;
        int searched_index;
        MultiUniFactor* node; // assume only one node connects to it

        //can be accessed and should be maintained
        vector<pair<Cube22, int>> ever_nnz_Y_bar; // <y_bar, index>
        Float* sum_Y_bar; // sum_Y_bar(k) = \sum_{k2} (Y_bar^01(k2, k) + Y_bar^{11}(k2, k) + Y_bar^10(k, k2) + Y_bar^{11}(k, k2))
        

        MultiBiFactor(int _K, MultiUniFactor* _node, ScoreVec* sv, Param* param){
            K = _K;
            c = sv->c;
            rho = param->rho;
            eta = param->eta;
            nnz_tol = param->nnz_tol;
            node = _node;
            node->edge = ((MultiBiFactor*)this);
            
            Y = new Cube22[K*K];
            for (int kk = 0; kk < K*K; kk++){
                Y[kk].Y00 = 0.0; Y[kk].Y01 = 0.0;
                Y[kk].Y10 = 0.0; Y[kk].Y11 = 0.0;
            }
            Y_bar = new Cube22[K*K];
            for (int kk = 0; kk < K*K; kk++){
                Y_bar[kk].Y00 = 0.0; Y_bar[kk].Y01 = 0.0;
                Y_bar[kk].Y10 = 0.0; Y_bar[kk].Y11 = 0.0;
            }
            sum_Y_bar = new Float[K];
            memset(sum_Y_bar, 0.0, sizeof(Float)*K);
            inside = new bool[K*K];
            memset(inside, 0, sizeof(bool)*K*K);
            is_ever_nnz = new bool[K*K];
            memset(is_ever_nnz, 0, sizeof(bool)*K*K);
        }

        ~MultiBiFactor(){
            delete[] Y;
            delete[] Y_bar;
            delete[] sum_Y_bar;
            delete[] is_ever_nnz;
            delete[] inside;
        }

        inline void naive_search(){
            Float* y_bar = node->y_bar;
            Float gmin = 0.0;
            int min_index = -1;
            for (int k1 = 0; k1 < K; k1++){
                for (int k2 = 0; k2 < K; k2++){
                    int k1k2 = k1*K+k2;
                    if (inside[k1k2]) continue;
                    Cube22& Ybar = Y_bar[k1k2];
                    Float g = c[k1k2] + rho * (Ybar.Y11*2 + Ybar.Y10 + Ybar.Y01  - y_bar[k1] - y_bar[k2]);
                    if (g < gmin){
                        gmin = g;
                        min_index = k1k2;
                    }
                }
            }
            searched_index = min_index;
            cerr << "bi_searched_index=" << searched_index << ", gmin=" << gmin << endl;
            if (min_index != -1){
                inside[min_index] = true;
                Cube22 cube;
                cube.Y00 = 0.0;
                cube.Y01 = 0.0;
                cube.Y10 = 0.0;
                cube.Y11 = 0.0;
                act_set.push_back(make_pair(cube, min_index));
            }
        }

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
            Cube22* Y_new = new Cube22[act_set.size()];

            int act_count = 0;
            if (fabs(A) < 1e-12){
				act_count = 0;
                for (vector<pair<Cube22, int>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
					int k1k2 = it->second;
					if (c[k1k2] <= 0){
						Y_new[act_count].Y11 = 1.0;
                        Y_new[act_count].Y10 = 0.0;
                        Y_new[act_count].Y01 = 0.0;
                        Y_new[act_count].Y00 = 0.0;
                    } else {
						Y_new[act_count].Y11 = 0.0;
                        Y_new[act_count].Y10 = 0.0;
                        Y_new[act_count].Y01 = 0.0;
                        Y_new[act_count].Y00 = 1.0;
                    }
				}
            } else {
                Float* b = new Float[4];
                Float* Y_new_cube = new Float[4];
                act_count = 0;
                Float* y_bar = node->y_bar;
                for (vector<pair<Cube22, int>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
					int k1k2 = it->second;
                    int k1 = k1k2 / K, k2 = k1k2 % K;
                    Cube22& cube = it->first;
                    Cube22 bar = Y_bar[k1k2];
                    Float ybar_k1 = y_bar[k1], ybar_k2 = y_bar[k2];
                    Float msgl = bar.Y10 + bar.Y11 - (cube.Y10 + cube.Y11) - ybar_k1;
                    Float msgr = bar.Y01 + bar.Y11 - (cube.Y01 + cube.Y11) - ybar_k2;
                    b[0] = cube.Y00;
                    b[1] = cube.Y01 - rho / A * msgr;
                    b[2] = cube.Y10 - rho / A * msgl;
                    b[3] = cube.Y11 - (rho * (msgl + msgr) + c[k1k2]) / A;

                    cerr << "b:\t" << b[0] << " " << b[1] << " " << b[2] << " " << b[3] << endl;

                    solve_simplex(4, Y_new_cube, b);
                    Y_new[act_count].Y11 = Y_new_cube[3];
                    Y_new[act_count].Y10 = Y_new_cube[2];
                    Y_new[act_count].Y01 = Y_new_cube[1];
                    Y_new[act_count].Y00 = Y_new_cube[0];
                }
                delete[] Y_new_cube;
            }

            //update Y_bar, sum_Y_bar, Y
            act_count = 0;
            for (vector<pair<Cube22, int>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
                int k1k2 = it->second;
                Cube22& old_cube = it->first;
                Cube22& new_cube = Y_new[act_count];
                Cube22& Ybar = Y_bar[k1k2];
                int k1 = k1k2 / K, k2 = k1k2 % K;
                //00
                Float delta_y = new_cube.Y00 - old_cube.Y00;
                Ybar.Y00 += delta_y;
                //01
                delta_y = new_cube.Y01 - old_cube.Y01;
                sum_Y_bar[k2] += delta_y;
                Ybar.Y01 += delta_y;
                //10
                delta_y = new_cube.Y10 - old_cube.Y10;
                sum_Y_bar[k1] += delta_y;
                Ybar.Y10 += delta_y;
                //11
                delta_y = new_cube.Y11 - old_cube.Y11;
                sum_Y_bar[k2] += delta_y;
                sum_Y_bar[k1] += delta_y;
                Ybar.Y11 += delta_y;

                it->first = new_cube;
                Y[k1k2] = new_cube;
            }

            //update ever_nnz_Y_bar 
            for (vector<pair<Cube22, int>>::iterator it = ever_nnz_Y_bar.begin(); it != ever_nnz_Y_bar.end(); it++){
                int k1k2 = it->second;
                it->first = Y_bar[k1k2];
            }
            if (searched_index != -1 && !is_ever_nnz[searched_index]){
                is_ever_nnz[searched_index] = true;
                Cube22 cube = Y_bar[searched_index];
                ever_nnz_Y_bar.push_back(make_pair(cube, searched_index));
            }

            delete[] Y_new;

            stats->bi_subsolve_time += get_current_time();            
        }

        //update Y_bar, sum_Y_bar
        inline void update_multipliers(){
            for (vector<pair<Cube22, int>>::iterator it = ever_nnz_Y_bar.begin(); it != ever_nnz_Y_bar.end(); it++){
                int k1k2 = it->second;
                Cube22& cube_y = Y[k1k2];
                Cube22& ybar = it->first;
                ybar.Y00 += eta*cube_y.Y00;
                ybar.Y10 += eta*cube_y.Y10;
                ybar.Y01 += eta*cube_y.Y01;
                ybar.Y11 += eta*cube_y.Y11;
                Y_bar[k1k2] = ybar; //= it->first;
                int k1 = k1k2 / K, k2 = k1k2 % K;
                sum_Y_bar[k1] += eta*(cube_y.Y10 + cube_y.Y11);
                sum_Y_bar[k2] += eta*(cube_y.Y01 + cube_y.Y11);
            }
        }
        
        inline void display(){

			cerr << endl;
            cerr << "Y:\t";
			for (vector<pair<Cube22, int>>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int kk = it->second;
                int k1 = kk / K, k2 = kk % K;
				cerr << "(" << k1 << "," << k2 << ")" << ":" << it->first.Y00 << "," << it->first.Y01 << "," << it->first.Y10 << "," << it->first.Y11 << " ";
			}
			cerr << endl;
            cerr << "Ybar:\t";
			for (vector<pair<Cube22, int>>::iterator it = ever_nnz_Y_bar.begin(); it != ever_nnz_Y_bar.end(); it++){
				int kk = it->second;
                int k1 = kk / K, k2 = kk % K;
				cerr << "(" << k1 << "," << k2 << ")" << ":" << it->first.Y00 << "," << it->first.Y01 << "," << it->first.Y10 << "," << it->first.Y11 << " ";
			}

		}
};
        
inline void MultiUniFactor::search(){
    stats->uni_search_time -= get_current_time();
    //compute gradient = c - rho \sum msg = c + K rho y_bar - rho \sum Y_bar
    for (int k = 0; k < K; k++){
        grad[k] = c[k] - rho * edge->sum_Y_bar[k];
    }
    for (vector<pair<Float, int>>::iterator uni_msg = ever_nnz_y_bar.begin(); uni_msg != ever_nnz_y_bar.end(); uni_msg++){
        Float ybar = uni_msg->first;
        int k = uni_msg->second;
        grad[k] += rho * ybar * K;
    }

    /*vector<pair<Cube22, int>>& ever_nnz_Y_bar = edge->ever_nnz_Y_bar;
      for (vector<pair<Cube22, int>>::iterator bi_msg = ever_nnz_Y_bar.begin(); bi_msg != ever_nnz_Y_bar.end(); bi_msg++){
      int k1k2 = bi_msg->second;
      Cube22& Y_bar = bi_msg->first;
      int k1 = k1k2 / K, k2 = k1k2 % K;
      grad[k1] -= rho*(Y_bar.Y10 + Y_bar.Y11);
      grad[k2] -= rho*(Y_bar.Y01 + Y_bar.Y11);
      }*/

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
    cerr << "uni_searched_index=" << searched_index << ", grad=" << -gmax << endl;
    if (max_index != -1){
        act_set.push_back(make_pair(0.0, max_index));
        inside[max_index] = true;
    }
    stats->uni_search_time += get_current_time();
}

inline void MultiUniFactor::subsolve(){
    if (act_set.size() == 0)
        return;
    stats->uni_subsolve_time -= get_current_time();
    Float* y_new = new Float[act_set.size()];
    int act_count = 0;
    if (edge == NULL || fabs(rho) < 1e-12){
        //min_y <c, y>, no bigram at all.
        cerr << "degenerated uni_subsolve!!!!!" << endl;
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
            cache[k] = sum_Y_bar[k] - c[k]/rho;
        }

        /*act_count = 0;
          vector<pair<Cube22, int>>& ever_nnz_Y_bar = edge->ever_nnz_msg;
          for (vector<pair<Cube22, int>>::iterator bi_msg = ever_nnz_Y_bar.begin(); bi_msg != ever_nnz_Y_bar.end(); bi_msg++){
          int k1k2 = bi_msg->second;
          Cube22& Ybar = bi_msg->first;
          int k1 = k1k2 / K, k2 = k1k2 % K;
          cache[k1] += (Ybar.Y10 + Ybar.Y11);
          cache[k2] += (Ybar.Y01 + Ybar.Y11);
          }*/

        act_count = 0;
        for (vector<pair<Float, int>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
            int k = it->second;
            cache[k] /= 2;
            cache[k] -= (y_bar[k] - it->first);
            if (cache[k] > 1.0)
                cache[k] = 1.0;
            if (cache[k] < 0.0)
                cache[k] = 0.0;
            y_new[act_count] = cache[k];
        }
    }

    //update y_bar, y

    act_count = 0;
    for (vector<pair<Float, int>>::iterator it = ever_nnz_y_bar.begin(); it != ever_nnz_y_bar.end(); it++, act_count++){
        int k = it->second;
        Float delta_y = y_new[act_count] - y[k];
        it->first += delta_y;
        y_bar[k] = it->first;
    }

    vector<pair<Float, int>> next_act_set;
    next_act_set.clear();
    act_count = 0;
    for (vector<pair<Float, int>>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
        int k = it->second;
        it->first = y_new[act_count];
        y[k] = it->first;
        //shrink
        if (it->first >= nnz_tol ){
            //this index is justed added by search()
            //Only if this index is active after subsolve, then it's added to ever_nnz_y_bar
            if (k == searched_index && !is_ever_nnz[k]){
                cerr << "y[k]=" << it->first << endl;
                ever_nnz_y_bar.push_back(make_pair(it->first, k));
                y_bar[k] = it->first;
                is_ever_nnz[k] = true;
                //adding_ever_nnz_msg(k);
            }
            next_act_set.push_back(make_pair(it->first, k));
        } else {
            inside[k] = false;
        }
    }
    act_set = next_act_set;

    delete[] y_new;
    stats->uni_subsolve_time += get_current_time();
}

inline Float MultiUniFactor::dual_inf(){
    for (int k = 0; k < K; k++){
        grad[k] = c[k] - rho * edge->sum_Y_bar[k];
    }
    for (vector<pair<Float, int>>::iterator uni_msg = ever_nnz_y_bar.begin(); uni_msg != ever_nnz_y_bar.end(); uni_msg++){
        Float ybar = uni_msg->first;
        int k = uni_msg->second;
        grad[k] += rho * ybar * K;
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
    return gmax;
}
#endif
