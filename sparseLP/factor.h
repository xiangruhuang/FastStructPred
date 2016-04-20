//prediction

#include "util.h"
#include "model.h"

//unigram factor, follows simplex constraints
class uni_factor{
	public:
	Float* alpha;
	int K;
	int m;
	vector<Float*> msgs;
	SparseVec* feature;
	Float** w; // w_k = w[:][k]
	Float* c; // c[k] = <w_k, x>
	Float* grad;
	Float* y;
	vector<int> act_set;
	bool* inside;
	Float rho, eta;

	uni_factor(int _K, SparseVec* _feature, Float** _w){
		K = _K;
		alpha = new Float[K];
		feature = _feature;
		w = _w;

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
	}

	~uni_factor(){
		delete[] alpha;
		delete[] c;
		delete[] y;
		delete[] grad;
		delete[] inside;
		edges.clear();
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
			act_set.push_back(k);
			inside[k] = true;
		}
	}

	void subsolve(){
		//	min_{y \in simplex} <c, y> + \rho/2 \sum_{msg \in msgs} \| (msg + y) - y \|_2^2
		// <===>min_{y \in simplex} \| y - 1/|msgs| ( \sum_{msg \in msgs} (msg + y) - 1/\rho c ) \|_2^2
		// <===>min_{y \in simplex} \| y - b \|_2^2
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
				msg[k] += delta_y;
			}
		}
		for (int k = 0; k < K; k++)
			y[k] = y_new[k];
		
		delete[] y_new;
		delete[] b;
	}


	//goal: minimize score
	Float score(){
		Float score = 0.0;
		for (int k = 0; k < K; k++){
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
	Float** y; // relaxed prediction matrix (vector)

	bi_factor(int _K, uni_factor* _l, uni_factor* _r, Float** _v){
		K = _K;
		l = _l;
		r = _r;
		v = _v;
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
		
		y = new Float*[K];
		for (int k = 0; k < K; k++){
			y[k] = new Float[K];
			memset(y[k], 0.0, sizeof(Float)*K);
		}
	}

	~bi_factor(){
		for (int k = 0; k < K; k++){
			delete[] beta[k];
		}
		delete[] beta;

		for (int k = 0; k < K; k++){
			delete[] y[k];
		}
		delete[] y;
		
		delete[] msgl;
		delete[] msgr;
		delete[] sumcol;
		delete[] sumrow;
	}
	
	void search(){
		
	}

	void subsolve(){

	}
	
	//     \mu^{t+1} - \mu^t
	//   = \msg^{t+1} - \msg^t
	//   = eta * ( (\sum_j y[:][j]) - y[:])
	void update_multipliers(){
		//update msgl, msgr
		Float* y_l = l->y;
		Float* y_r = r->y;
		for (int k = 0; k < K; k++){
			msgl[k] += eta * (sumrow[k] - y_l[k])
			msgr[k] += eta * (sumcol[k] - y_r[k])
		}
	}

	Float score(){
		Float score = 0.0;
		for (int k1 = 0; k1 < K; k1++){
			for (int k2 = 0; k2 < K; k2++){
				score += v[k1][k2]*y[k1][k2];
			}
		}
		return score;
	}
};
