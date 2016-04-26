//prediction

#include "util.h"
#include "model.h"

//unigram factor, follows simplex constraints
class uni_factor{
	public:
	//fixed
	int K;
	Float rho;
	//Float eta;
	Float* c; // c[k] = -<w_k, x>
	
	//maintained
	Float* grad;
	Float* y;
	bool* inside;
	vector<Float*> msgs;
	vector<int> act_set;

	uni_factor(int _K, SparseVec* feature, Float** w, Param* param){
		K = _K;
		//feature = _feature;
		//w = _w;
		rho = param->rho;
		//eta = param->eta;

		//compute score vector
		c = new Float[K];
		memset(c, 0.0, sizeof(Float)*K);
		for (SparseVec::iterator it = feature->begin(); it != feature->end(); it++){
			Float* wj = w[it->first];
			Float x_j = it->second;
			for (int k = 0; k < K; k++){
				c[k] -= wj[k] * x_j;
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
		msgs.clear();

		//temporary
		//fill_act_set();
	}

	~uni_factor(){
		delete[] c;
		delete[] y;
		delete[] grad;
		delete[] inside;
		act_set.clear();
		msgs.clear();
	}
	
	void fill_act_set(){
		act_set.clear();
		for (int k = 0; k < K; k++){
			act_set.push_back(k);
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
		//cerr << "g[9]=" << grad[9] << ", gmax=" << gmax << endl;
		if (max_index != -1){
			act_set.push_back(max_index);
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
			if (y[k] > 1e-12 || grad[k] < 0.0){
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
		for (int k = 0; k < K; k++){
			score += c[k]*y[k];
		}
		return score;
	}

	// F = c^T y + \rho/2 \sum_{msg} \|msg \|^2
	Float func_val(){
		Float val = 0.0;
		for (int k = 0; k < K; k++){
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
		cerr << "act_set=" << act_set.size() << endl;
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
	int K;
	Float rho, eta;
	uni_factor* l;
	uni_factor* r;
	Float* c; // score vector: c[k1k2] = -v[k1k2/K][k1k2%K];
	
	//maintained
	Float* msgl; // message to uni_factor l
	Float* msgr; // messageto uni_factor r
	Float* sumcol; // sumcol[k] = sum(y[:][k])
	Float* sumrow; // sumrow[k] = sum(y[k][:])
	Float* y; // relaxed prediction matrix (vector)
	Float* grad;
	vector<int> act_set;
	bool* inside;

	bi_factor(int _K, uni_factor* _l, uni_factor* _r, Float** _v, Param* param){
		K = _K;
		l = _l;
		r = _r;
		rho = param->rho;
		eta = param->eta;

		msgl = new Float[K];
		memset(msgl, 0.0, sizeof(Float)*K);
		msgr = new Float[K];
		memset(msgr, 0.0, sizeof(Float)*K);
		l->msgs.push_back(msgl);
		r->msgs.push_back(msgr);
		sumcol = new Float[K];
		sumrow = new Float[K];
		memset(sumcol, 0.0, sizeof(Float)*K);
		memset(sumrow, 0.0, sizeof(Float)*K);
		c = new Float[K*K];
		for (int k1k2 = 0; k1k2 < K*K; k1k2++){
			c[k1k2] = -(_v[k1k2/K][k1k2%K]);
		}
		
		y = new Float[K*K];
		memset(y, 0.0, sizeof(Float)*K*K);
		
		//cache
		grad = new Float[K*K];
		memset(grad, 0.0, sizeof(Float)*K*K);
		inside = new bool[K*K];
		memset(inside, false, sizeof(bool)*K*K);
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
		delete[] grad;
		delete[] inside;
		act_set.clear();
	}
	
	void fill_act_set(){
		act_set.clear();
		for (int kk = 0; kk < K*K; kk++){
			act_set.push_back(kk);
			inside[kk] = true;
		}
	}
	
	inline void search(){
		//compute gradient
	
		//find argmax
		Float gmax = 0.0;
		int max_index = -1;
		for (Int k1k2 = 0; k1k2 < K * K; k1k2++){
			if (inside[k1k2]) continue;
			int k1=k1k2/K, k2 = k1k2%K;
			Float g = c[k1k2] + rho * (msgl[k1] + msgr[k2]);
			if (-g > gmax){
				gmax = -g;
				max_index = k1k2;
			}
		}
		//cerr << "msgl[27]=" << msgl[27] << endl;
		//cerr << "g[27,9] = " << (c[27*K+9] + rho * (msgl[27] + msgr[9])) << ", gmax = " << gmax << ", max(k1,k2)=" << (max_index / K) << "," << (max_index % K)<< endl;
		if (max_index != -1){
			act_set.push_back(max_index);
			inside[max_index] = true;
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
		int* sl = new int[K];
		int* sr = new int[K];
		memset(sl, 0, sizeof(int)*K);
		memset(sr, 0, sizeof(int)*K);
		int max_sl = 0, max_sr = 0;
		for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
			int k1k2 = *it;
			int k1 = k1k2 / K, k2 = k1k2 % K;
			sl[k1]++; sr[k2]++;
			if (max_sl < sl[k1])
				max_sl = sl[k1];
			if (max_sr < sr[k2])
				max_sr = sr[k2];
		}
		
		Float A = rho * 2 * K; //(l->act_set.size() + r->act_set.size());
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
				int k1 = k1k2 / K;
				int k2 = k1k2 % K;
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
			int k1 = k1k2 / K, k2 = k1k2 % K;
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
		for (int k = 0; k < K; k++){
			msgl[k] += eta * (sumrow[k] - y_l[k]);
			msgr[k] += eta * (sumcol[k] - y_r[k]);
			if (fabs(msgr[k]) > 1e-12){
				//cerr << "k=" << k << ", msgr[k]=" << msgr[k] << endl;
			}
		}
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
		for (int k = 0; k < K*K; k++){
			val += c[k]*y[k];
		}
		for (int k = 0; k < K; k++){
			val += rho/2 * msgl[k] * msgl[k];
			val += rho/2 * msgr[k] * msgr[k];
		}
		return val;
	}
	
	Float infea(){
		Float p_inf = 0;
		Float* y_l = l->y;
		Float* y_r = r->y;
		for (int k = 0; k < K; k++){
			p_inf += fabs(sumrow[k] - y_l[k]);
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
			int k = *it;
			cerr << "(" << (k/K) << "," << (k%K) << ")" << ":" << y[k] << " ";
		}
		cerr << endl;
	}

};
