//prediction

#include "util.h"
#include "model.h"

//unigram factor, follows simplex constraints
class uni_factor{
	public:
	//fixed
	int K;
	Float rho, eta;
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
		eta = param->eta;

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
		for (int k = 0; k < K; k++){
			act_set.push_back(k);
			inside[k] = true;
		}
	}

	~uni_factor(){
		delete[] c;
		delete[] y;
		delete[] grad;
		delete[] inside;
		act_set.clear();
		msgs.clear();
	}
	
	void search(){
		//compute gradient of y_i
		for (int k = 0; k < K; k++){
			grad[k] = c[k];
		}
		for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
			Float* msg = *m;
			for (int k = 0; k < K; k++)
				grad[k] += eta * msg[k];
		}
		Float gmax = -1e300;
		int max_index = -1;
		for (int k = 0; k < K; k++){
			if (inside[k]) continue;
			//if not inside, y_k is guaranteed to be zero, and y_k >= 0 by constraint
			if (grad[k] > 0.0) continue;
			if (fabs(grad[k]) > gmax){
				gmax = fabs(grad[k]);
				max_index = k;
			}
		}
		if (max_index != -1){
			act_set.push_back(max_index);
			inside[max_index] = true;
		}
	}


	//	min_{y \in simplex} <c, y> + \rho/2 \sum_{msg \in msgs} \| (msg + y) - y \|_2^2
	// <===>min_{y \in simplex} \| y - 1/|msgs| ( \sum_{msg \in msgs} (msg + y) - 1/\rho c ) \|_2^2
	// <===>min_{y \in simplex} \| y - b \|_2^2	
	// no shrinking now
	void subsolve(){
		if (msgs.size() == 0){
			//min_y <c, y>
			Float cmin = 1e300;
			int min_index = -1;
			for (int k = 0; k < K; k++){
				if (c[k] < cmin){
					cmin = c[k];
					min_index = k;
				}
			}
			assert(min_index != -1 && "smallest coordinate should exist");
			memset(y, 0.0, sizeof(Float)*K);
			y[min_index] = 1.0;
			return;
		}
		Float* b = new Float[K];
		Float* y_new = new Float[K];
		memset(b, 0.0, sizeof(Float)*K);
		for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
			Float* msg = *m;
			for (int k = 0; k < K; k++){
				b[k] += msg[k] + y[k];
			}
		}
		int n = msgs.size();
		for (int k = 0; k < K; k++){
			b[k] -= 1/rho * c[k];
			b[k] /= (Float)n;
		}
		
		solve_simplex(K, y_new, b);
		for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
			Float* msg = *m;
			for (int k = 0; k < K; k++){
				Float delta_y = y_new[k] - y[k];
				msg[k] -= delta_y; // since msg = M Y - y + \mu
			}
		}
		for (int k = 0; k < K; k++){
			assert(!isnan(y_new[k]) && "y_new[k] not a number?");
			y[k] = y_new[k];
		}
		
		delete[] y_new;
		delete[] b;
	}

	//goal: minimize score
	Float score(){
		Float score = 0.0;
		for (int k = 0; k < K; k++){
			assert(!isnan(c[k]) /* c[k] not a number */);
			assert(!isnan(y[k]) /* y[k] not a number */);
			score += c[k]*y[k];
		}
		return score;
	}
};

//bigram factor
class bi_factor{
	public:
	Float** beta;
	int K;
	uni_factor* l;
	uni_factor* r;
	Float* msgl; // message to uni_factor l
	Float* msgr; // messageto uni_factor r
	Float* sumcol; // sumcol[k] = sum(y[:][k])
	Float* sumrow; // sumrow[k] = sum(y[k][:])
	Float** v; // v_{k_1, k_2} = v[k_1][k_2], k_1 for l, k_2 for r
	Float* y; // relaxed prediction matrix (vector)
	Float* grad;
	vector<int> act_set;
	Float rho, eta;

	bi_factor(int _K, uni_factor* _l, uni_factor* _r, Float** _v, Param* param){
		K = _K;
		l = _l;
		r = _r;
		v = _v;
		rho = param->rho;
		eta = param->eta;

		beta = new Float*[K];
		for (int k = 0; k < K; k++)
			beta[k] = new Float[K];

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
		
		y = new Float[K*K];
		memset(y, 0.0, sizeof(Float)*K*K);
		
		//cache
		grad = new Float[K*K];
		memset(grad, 0.0, sizeof(Float)*K*K);
		act_set.clear();
	}

	~bi_factor(){
		for (int k = 0; k < K; k++){
			delete[] beta[k];
		}
		delete[] beta;

		delete[] y;
		
		delete[] msgl;
		delete[] msgr;
		delete[] sumcol;
		delete[] sumrow;
		delete[] grad;
		act_set.clear();
	}
	
	void search(){
		
		//compute gradient
	
		//find argmax
		Float gmax = -1e300;
		int max_index = -1;
		for (Int k1k2 = 0; k1k2 < K * K; k1k2++){
			if (grad[k1k2] > gmax){
				
			}
		}
		if (max_index != -1){
			act_set.push_back(max_index);
		}
	}

	//       min_Y  <Y, v> + \rho/2 ( \| msgl \|_2^2 + \| msgr \|_2^2 )
	// <===> min_Y  \sum_{(k_1, k_2)} [ A/2 Y^2_{(k_1, k_2)} + B_{(k_1, k_2)} Y_{(k_1, k_2)} ]
	//        s.t. 	A = 2 * \rho
	// 		B_{(k_1, k_2)} = 2 * (msgl[k_1] + msgr[k_2] - Y_{(k_1, k_2)})
	// 		0 <= Y <= 1
	void subsolve(){
		Float A = 2 * rho;
		//compute gradient
		
		//for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
		//	int k1k2 = *it;
		for (int k1k2 = 0; k1k2 < K * K; k1k2++){
			int k1 = k1k2 / K;
			int k2 = k1k2 % K;
			grad[k1k2] = v[k1][k2] + rho * (msgl[k1] + msgr[k2]);
		}

		//compute Dk
		Float* Dk = new Float[K*K];
		for (int k1k2 = 0; k1k2 < K * K; k1k2++){
			Dk[k1k2] = grad[k1k2] + A * 1;
		}

		sort(Dk, Dk + K*K, greater<Float>() );

		Float b = Dk[0] - A*1;
		Int r;
		for( r=1; r<K*K && b<r*Dk[r]; r++)
			b += Dk[r];
		b = b / r;
		
		Float* y_new = new Float[K*K];
		//record alpha new values
		for(Int k1k2=0;k1k2<K*K;k1k2++){
			assert(false /* not implemented yet */);
			y_new[k1k2] = 0; // TODO //min( (Float)((k1k2!=yi_yj)?0.0:1.0), (b-grad[k1k2])/A );
		}
		//update y_new
		for (int k1k2 = 0; k1k2 < K * K; k1k2++){
			Float delta_y = y_new[k1k2] - y[k1k2];
			int k1 = k1k2 / K, k2 = k1k2 % K;
			msgl[k1] += delta_y; sumrow[k1] += delta_y;
			msgr[k2] += delta_y; sumcol[k2] += delta_y;
		}
		
		for(Int k1k2=0;k1k2<K*K;k1k2++){
			y[k1k2] = y_new[k1k2];
		}
		
		delete[] y_new;
		delete[] Dk;
	}
	
	//     \mu^{t+1} - \mu^t
	//   = \msg^{t+1} - \msg^t
	//   = eta * ( (\sum_j y[:][j]) - y[:])
	void update_multipliers(){
		//update msgl, msgr
		Float* y_l = l->y;
		Float* y_r = r->y;
		for (int k = 0; k < K; k++){
			msgl[k] += eta * (sumrow[k] - y_l[k]);
			msgr[k] += eta * (sumcol[k] - y_r[k]);
		}
	}

	Float score(){
		Float score = 0.0;
		for (int k1 = 0; k1 < K; k1++){
			int k1K = k1*K;
			Float* v_k1 = v[k1];
			for (int k2 = 0; k2 < K; k2++){
				score += v_k1[k2]*y[k1K+k2];
			}
		}
		return score;
	}
};
