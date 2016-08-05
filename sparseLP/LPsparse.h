#ifndef LP_SPARSE
#define LP_SPARSE

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include "LPutil.h"
//#include "model.h"
//#include "CG.h"
#include "problem.h"
#include "util.h"

using namespace std;

extern double prediction_time;

class LP_Param{

	public:

		char* data_dir;

		bool solve_from_dual;
		bool use_CG;

		double tol;
		double tol_trans;
		double tol_sub;

		double eta;
		double nnz_tol;
		int max_iter;

		LP_Param(){
			solve_from_dual = false;
			use_CG = false;

			tol = 1e-5;
			//tol_trans = 10*tol;
			tol_trans = -1;
			tol_sub = 0.5*tol_trans;

			eta = 1;
			nnz_tol = 1e-4;
			max_iter = 100000;
		}
};

//this heap maintains gradient of candidates of active set (currently not active), thus x_j = 0 \forall j
// gradient_j = c[j] + eta_t(<A^I_t[j], [w^I]_+>, <A^E_t[j], w^E> )
//for j <= n, there is no constraint on x_j, thus we only care fabs(gradient)
//for j > n, x_j >= 0, (and here we have x_j == 0), thus we need gradient < 0
/*inline void maintain_heap(int n, int m, int index, Float delta_wj, Constr* A, ConstrInv* At, Heap heap){

  if (index < m){
//inside w^I

}
}*/

/** Using Randomized Coordinate Descent to solve:
 *
 *  min c'x + \frac{eta_t}{2}\| (Ax-b+alpha_t/eta_t)_+ \|^2 + \frac{eta_t}{2}\|Aeq*x-t*beq+beta_t/eta_t\|^2
 *  s.t. x >= 0
 */
void rcd(int n, int nf, int m, int me, ConstrInv* At, double* b, double* c, double* x, double* w, double* h2_jj, double* hjj_ubound, double eta_t, int& niter, int inner_max_iter, int& active_matrix_size, double& PGmax_old_last, int phase, LP_Param* param){
	int max_num_linesearch = 20;
	double sigma = 0.01; //for line search, find smallest t \in {0,1,2,3...} s.t.
	// F(x+beta^t*d)-F(x) <= sigma * beta^t g_j*d

	//initialize active index
	int* index = new int[n+nf];
	int active_size = n+nf;
	for (int i = 0; i < n+nf; i++)
		index[i] = i;

	int iter=0;
	double PG, PGmax_old = 1e300, PGmax_new;
	double d;
	bool* inside = new bool[n + nf];
	memset(inside, 0, sizeof(bool)*(n+nf));

	//ArrayHeap heap = new pair<Float, Int>[n+nf];


	while(iter < inner_max_iter){
		PGmax_new = -1e300;
		random_shuffle(index, index+active_size);

		for(int s=0;s<active_size;s++){

			int j = index[s];

			//compute gradient, hessian of j-th coordinate
			double g = 0.0;
			double hjj = 0.0;
			for(ConstrInv::iterator it=At[j].begin(); it!=At[j].end(); it++){
				if( it->first < m && w[it->first] > 0.0 ){
					g += w[it->first] * it->second;
					hjj += it->second * it->second;
				}
				else if( it->first >= m ){
					g += w[it->first] * it->second;
				}
			}

			g *= eta_t;
			g += c[j];
			hjj *= eta_t;
			hjj += h2_jj[j];
			hjj = max( hjj, 1e-3);
			//hjj += 1.0;

			//compute PG and determine shrinking
			if( j>=n || x[j] > 0.0 ){

				PG = fabs(g);
			}else{
				PG = 0.0;
				//if( g > PGmax_old ){
				if( g > 0 ){
					active_size--;
					swap( index[s], index[active_size] );
					s--;
					continue;
				}else if( g < 0.0 ){
					PG = fabs(g);
				}
			}
			//cerr << "PG=" << PG << endl;

			if( PG > PGmax_new )
				PGmax_new = PG;
			if( PG < 1e-12 )
				continue;
			//compute d = Delta x
			if( j < n )
				d = max(x[j]-g/hjj, 0.0)-x[j];
			else
				d = -g/hjj;
			//cerr << "d=" << d << endl;

			//line search
			double d_old = 0.0, d_diff, rhs, cond, appxcond, fnew, fold;
			double delta = g*d;
			int t;
			for(t=0; t<max_num_linesearch; t++){

				d_diff = d - d_old;

				cond = -sigma*delta;
				appxcond = hjj_ubound[j]*d*d/2.0 + g*d + cond;
				//cerr << "appxcond=" << appxcond << endl;
				if( appxcond <= 0.0 ){
					//update w, v
					for(ConstrInv::iterator it=At[j].begin();it!=At[j].end();it++)
						w[it->first] += d_diff * it->second;
					break;
				}

				if( t == 0 ){
					//compute fold, fnew (related to coordinate j)
					fold = c[j] * x[j];
					fnew = c[j] * (x[j] + d);
					double tmp_old=0.0, tmp_new=0.0;
					for(ConstrInv::iterator it=At[j].begin();it!=At[j].end();it++){
						//fold
						if( it->first >= m || w[it->first] > 0.0 )
							tmp_old += w[it->first]*w[it->first];
						//update w
						w[it->first] += d_diff * it->second;
						//fnew
						if( it->first >= m || w[it->first] > 0.0 )
							tmp_new += w[it->first]*w[it->first];
					}
					fold += eta_t*tmp_old/2.0;
					fnew += eta_t*tmp_new/2.0;

				}else{
					fnew = c[j] * (x[j]+d);
					double tmp = 0.0;
					for(ConstrInv::iterator it=At[j].begin();it!=At[j].end();it++){
						//update w
						w[it->first] += d_diff * it->second;
						if( it->first >= m || w[it->first] > 0.0 )
							tmp += w[it->first]*w[it->first];
					}
					fnew += eta_t*tmp/2.0;
				}

				cond += fnew - fold;
				if( cond <= 0 )
					break;
				else{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}
			if( t == max_num_linesearch ){
				//cerr << "reach max_num_linesearch" << endl;
				return;
			}
			//update x_j
			x[j] += d;
			//cerr << "updating j=" << j << ", d=" << d << ", x^+=" << x[j] << endl;	
			}
			iter++;
			if( iter % 10 == 0 ){
				//cerr << "iter=" << iter << ", act=" << active_size << "/" << (n+nf) << ", gmax=" << PGmax_new << endl;
			}

			PGmax_old = PGmax_new;
			if( PGmax_old <= 0.0 )
				PGmax_old = 1e300;

			if( PGmax_new <= param->tol_sub || iter == inner_max_iter ){ //reach stopping criteria

				//cerr << "\tinner rcd iter=" << iter << ", act=" << active_size << "/" << (n+nf) << ", gmax=" << PGmax_new << endl;
				break;
			}
		}
		//passing some info
		niter = iter;
		active_matrix_size = 0;
		for(int s=0;s<active_size;s++)
			active_matrix_size += At[index[s]].size();
		PGmax_old_last = PGmax_old;

		delete[] inside;
		delete[] index;
	}


	/** Solve:
	 *
	 *  min  c'x
	 *  s.t. Ax <= b
	 *       Aeq x = beq
	 *       x >= 0
	 */
	void LPsolve(int n, int nf, int m, int me, Constr* A, ConstrInv* At, double* b, double* c, double*& x, double*& w, LP_Param* param){

		double eta_t = param->eta;
		int max_iter = param->max_iter;

		for(int j=0;j<n+nf;j++)
			x[j] = 0.0;
		//w_t=Ax-b+w_{t-1}, v=Aeq*x-beq+ v_{t-1}
		for(int i=0;i<m+me;i++)
			w[i] = -b[i]; //w=y/eta_t ==> w=-b+y/eta_t

		//initialize h2_ii (H=H1+H2) 
		double* h2_jj = new double[n+nf];
		for(int j=0;j<n+nf;j++)
			h2_jj[j] = 0.0;
		for(int i=m;i<m+me;i++){
			for(Constr::iterator it=A[i].begin(); it!=A[i].end(); it++){
				h2_jj[it->first] += it->second*it->second;
			}
		}
		for(int j=0;j<n+nf;j++)
			h2_jj[j] *= eta_t;

		//initialize hjj's upper bound
		double* hjj_ubound = new double[n+nf];
		for(int j=0;j<n+nf;j++)
			hjj_ubound[j] = 0.0;
		for(int i=0;i<m;i++){
			for(Constr::iterator it=A[i].begin(); it!=A[i].end(); it++){
				hjj_ubound[it->first] += it->second*it->second;
			}
		}
		for(int j=0;j<n+nf;j++){
			hjj_ubound[j] *= eta_t;
			hjj_ubound[j] += h2_jj[j];
		}
		//calculate matrix total # of nonzeros entries
		int nnz_A = 0;
		for(int i=0;i<n+nf;i++){
			nnz_A += At[i].size();
		}

		//main loop 
		double minus_time = 0.0;
		int inner_max_iter=1;
		double PGmax_old = 1e300;
		int niter;
		int print_per_iter = 5;
		double pinf = 1e300, gap=1e300, dinf=1300, obj; //primal infeasibility
		int nnz;//, nnz_last=1;
		int active_nnz = nnz_A;
		int phase = 1, inner_iter;
		double dinf_last=1e300, pinf_last=1e300, gap_last=1e300;
		for(int t=0;t<max_iter;t++){
			if( phase == 1 ){
				inner_max_iter = (active_nnz!=0)?(nnz_A/active_nnz):(n+nf);
				//inner_max_iter = 10000;
				rcd(n,nf,m,me, At,b,c,  x, w, h2_jj, hjj_ubound, eta_t, niter, inner_max_iter, active_nnz, PGmax_old, phase, param);

			}else if( phase == 2 ){

				rcd(n,nf,m,me, At,b,c,  x, w, h2_jj, hjj_ubound, eta_t, niter, inner_max_iter, active_nnz, PGmax_old, phase, param);

				//cerr << endl;
			}
			//niter = inner_iter;

			if( t % print_per_iter ==0 ){

				nnz = 0;
				for(int j=0;j<n+nf;j++)
					if( x[j] > param->nnz_tol )
						nnz++;

				obj = 0.0;
				for(int j=0;j<n+nf;j++)
					obj += c[j]*x[j];

				pinf = primal_inf(n,nf,m,me,x,A,b);
				dinf = dual_inf(n,nf,m,me,w,At,c,eta_t);
				gap = duality_gap(n,nf,m,me,x,w,c,b,eta_t);

				cerr << setprecision(7) << "iter=" << t << ", #inner=" << niter << ", obj=" << obj << ", eta=" << eta_t << ", time=" << ((double)get_current_time() + prediction_time);
				cerr << setprecision(2) << ", p_inf=" << pinf << ", d_inf=" << dinf << ", gap=" << fabs(gap/obj) << ", nnz=" << nnz << "(" << ((double)active_nnz/nnz_A) << ")" ;
				cerr << endl;
			}

			if( obj <= 0 && pinf<=param->tol && dinf<=param->tol ){
				//cerr << "iter=" << t << ", nnz=" << nnz << ", pinf=" << pinf << ", dinf=" << dinf;
				break;
			}

			// w_t = Ax-b + w_{t-1} = w_t-1 + alpha_{t}/eta_t - alpha_{t-1}/eta_t  

			for(int i=0;i<m;i++){ //inequality
				if( w[i] > 0 )
					w[i] -= b[i];
				else
					w[i] = -b[i];
			}
			for(int i=m;i<m+me;i++){ //equality
				w[i] -= b[i];
			}

			for(int j=0;j<n+nf;j++){ //both equality & inequality
				double tmp = x[j];
				for(ConstrInv::iterator it=At[j].begin(); it!=At[j].end(); it++)
					w[it->first] += tmp * it->second;
			}

			if( phase == 1 && pinf <= param->tol_trans ){

				phase = 2;
				print_per_iter = 1;
				//cerr << "phase = 2" << endl;
				inner_max_iter = 100;
			}

			if( phase == 2 ){

				if( (niter < 2) || (pinf < 0.5*dinf && dinf>dinf_last) ){

					if(niter < inner_max_iter) param->tol_sub *= 0.5;
					else			   inner_max_iter = min(inner_max_iter*2, 10000);

				}

				if( dinf < 0.5*pinf && pinf > pinf_last ){

					//////////////////////////////////////correction
					for(int i=0;i<m;i++){ //inequality
						if( w[i] > 0 )
							w[i] -= b[i];
						else
							w[i] = -b[i];
					}
					for(int i=m;i<m+me;i++){ //equality
						w[i] -= b[i];
					}

					for(int j=0;j<n+nf;j++){ //both equality & inequality
						double tmp = x[j];
						for(ConstrInv::iterator it=At[j].begin(); it!=At[j].end(); it++)
							w[it->first] += tmp * it->second;
					}
					//////////////////////////////////////////////
					eta_t *= 2;
					for(int j=0;j<n+nf;j++){
						h2_jj[j] *= 2;
						hjj_ubound[j] *= 2;
					}
					for(int i=0;i<m+me;i++)
						w[i] /= 2;
				}

			}
			dinf_last = dinf;
			pinf_last = pinf;
			gap_last = gap;
		}
		param->eta = eta_t;
	}

	//a template/wrapper used to cache shared parts of each sample, such as A & At
	class LP_Problem{
		public:
			int n; // #(unconstrainted x)
			int nf; // #(x >= 0)
			int m; // number of inequality constraints 
			int me; // number of equality constraints
			Constr* A; //m+me by n+nf
			double* b; 
			Constr* At; //n+nf by m+me
			double* c; 

			double* x;
			double* y;
			LP_Param* param;

			LP_Problem(){
				param = new LP_Param();
			}

			LP_Problem(Param* _param){
				param = new LP_Param();
				param->eta = _param->rho;
				param->nnz_tol = _param->nnz_tol;
			}

			void solve(){
				LPsolve(n, nf, m, me, A, At, b, c, x, y, param);
			}

			~LP_Problem(){
				for (int i = 0; i < m+me; i++)
					A[i].clear();
				delete[] A;
				for (int i = 0; i < n+nf; i++)
					At[i].clear();
				delete[] At;
				delete[] b;
				delete[] c;
				delete[] x;
				delete[] y;
			}
	};

	LP_Problem* construct_LP_multi(Instance* ins, Param* param){
		LP_Problem* ins_pred_prob = new LP_Problem(param);
		int K = ins->node_label_lists[0]->size();
		int M = K*(K-1)/2; //number of bigram factor
		int m = 0, nf=0, n=K+M*4, me=2*M + M; //consistency + simplex

		ins_pred_prob->m = m;
		ins_pred_prob->nf = nf;
		ins_pred_prob->n = n;
		ins_pred_prob->me = me;

		Constr* A = new Constr[m+me];
		double* b = new double[m+me];
		double* c = new double[n+nf];

		//unigram 
		//cerr << "c:\t";
		for(Int i=0;i<K;i++){
			c[i] = ins->node_score_vecs[0][i];
			//  cerr << c[i] << " ";
		}
		//cerr << endl;

		/*if( y != NULL ){ //loss-augmented decoding
		  for(Int k=0; k<K; k++){
		  if( find( y->begin(), y->end(), k ) == y->end() )
		  c[k] -= 1.0;
		  else
		  c[k] -= -1.0;
		  }
		  }*/

		//cerr << "|||| ";
		//bigram
		for(Int i=K;i<K+M*4;i++)
			c[i] = 0.0;
		int ij4=0;
		for(Int i=0;i<K; i++){
			for(Int j=i+1; j<K; j++, ij4+=4){
				c[ K + ij4 + 3 ] = ins->edge_score_vecs[0]->c[i*K+j];
				//cerr << c[K+ij4+3] << " ";
			}
		}
		//cerr << endl;

		int row=0;
		ij4 = 0;
		//construct consistency constraint
		for(int i=0;i<K;i++){ //for each bigram factor
			for(int j=i+1; j<K; j++, ij4+=4){
				//right consistency
				/*A[row].push_back(make_pair((K+ij4+2*0+1*0), 1.0));
				  A[row].push_back(make_pair((K+ij4+2*1+1*0), 1.0));
				  A[row].push_back(make_pair(j, 1.0)); //b00+b10=1-u1
				  b[row++] = 1.0;
				 */
				A[row].push_back(make_pair((K+ij4+2*0+1*1), 1.0));
				A[row].push_back(make_pair((K+ij4+2*1+1*1), 1.0));
				A[row].push_back(make_pair(j, -1.0)); //b01+b11=u1
				b[row++] = 0.0;
				//left consistency
				/*A[row].push_back(make_pair((K+ij4+2*0+1*0), 1.0));
				  A[row].push_back(make_pair((K+ij4+2*0+1*1), 1.0));
				  A[row].push_back(make_pair(i, 1.0)); //b00+b01=1-u1
				  b[row++] = 1.0;
				 */
				A[row].push_back(make_pair((K+ij4+2*1+1*0), 1.0));
				A[row].push_back(make_pair((K+ij4+2*1+1*1), 1.0));
				A[row].push_back(make_pair(i, -1.0));//b10+b11=u1
				b[row++] = 0.0;
			}
		}

		ij4=0;
		for(int i=0;i<K;i++){
			for(int j=i+1; j<K; j++, ij4+=4){

				for(int d=0;d<4;d++)
					A[row].push_back(make_pair((K+ij4+d), 1.0)); //simplex
				b[row++] = 1.0;
			}
		}

		assert( row == m+me );

		ConstrInv* At = new ConstrInv[n+nf];
		transpose(A, m+me, n+nf, At);
		ins_pred_prob->A = A;
		ins_pred_prob->b = b;
		ins_pred_prob->c = c;
		ins_pred_prob->At = At;

		ins_pred_prob->x = new double[n+nf];
		ins_pred_prob->y = new double[m+me];

		return ins_pred_prob;

	}

	LP_Problem* construct_LP(Instance* ins, Param* param){
		if (param->problem_type == "multilabel"){
			return construct_LP_multi(ins, param);
		}
		LP_Problem* ins_pred_prob = new LP_Problem(param);
		ins_pred_prob->param->eta = param->rho;
		int T = ins->T;

		//for i-th unigram factor(numbered i), we have K variables, numbered K*i+{0..K-1}
		//for i-th bigram factor( (T+i)-th factor ), we have K*K variables, numbered T*K + K*K*i+(k1*K+k2)
		int m = 0; // no inequality
		int nf = 0; // no unconstrainted x
		int n = 0; //T*K + (T - 1)*K*K; // number of non-negative variables
		int me = 0; //T + ins->edges.size() * (2 * K); // number of equality constraints

		for (int i = 0; i < ins->T; i++){
			n += ins->node_label_lists[i]->size();
			me++;
		}

		for (vector<pair<int, int>>::iterator e = ins->edges.begin(); e != ins->edges.end(); e++){
			int i = e->first, j = e->second;
			int K1 = ins->node_label_lists[i]->size();
			int K2 = ins->node_label_lists[j]->size();
			n += K1*K2;
			me += (K1+K2);
		}

		ins_pred_prob->m = m;
		ins_pred_prob->nf = nf;
		ins_pred_prob->n = n;
		ins_pred_prob->me = me;

		Constr* A = new Constr[m+me];
		double* b = new double[m+me];
		double* c = new double[n+nf];
		int row = 0;

		//construct constraints of unigram factors
		int* offset = new int[T + ins->edges.size() + 1];
		offset[0] = 0;
		for (int i = 0; i < T; i++){
			int K = ins->node_label_lists[i]->size();
			// \sum_{k=0}^{K} alpha_i(k) = 1.0
			for (int k = 0; k < K; k++){
				A[row].push_back(make_pair(offset[i]+k, 1.0));
			}
			b[row++] = 1.0;
			offset[i+1] = offset[i] + K;
		}

		//construct constraints of bigram factors
		int edge_count = 0;
		for (vector<pair<int, int>>::iterator e = ins->edges.begin(); e != ins->edges.end(); e++, edge_count++){
			int i = e->first, j = e->second;
			//we are inside bigram factor (i,j), assume (i < j)
			//assert(i < j && j < T /* node id should be in [T] */);
			cerr << edge_count << "/" << ins->edges.size() << endl;

			//for each k1, alpha_i(k1) = \sum_{k2=0}^K beta_{(i,j)}(k1, k2)
			int K1 = ins->node_label_lists[i]->size();
			int K2 = ins->node_label_lists[j]->size();
			for (Int k1 = 0; k1 < K1; k1++){
				for (Int k2 = 0; k2 < K2; k2++){
					A[row].push_back(make_pair(offset[T+edge_count] + k1*K2 + k2, 1.0));
				}
				A[row].push_back(make_pair(offset[i]+k1, -1.0));
				b[row++] = 0.0;
			}

			//for each k2, alpha_j(k2) = \sum_{k1=0}^K beta_{(i,j)}(k1, k2)
			for (Int k2 = 0; k2 < K2; k2++){
				for (Int k1 = 0; k1 < K1; k1++){
					A[row].push_back(make_pair(offset[T+edge_count] + k1*K2 + k2, 1.0));
				}
				A[row].push_back(make_pair(offset[j]+k2, -1.0));
				b[row++] = 0.0;
			}
			offset[T + edge_count + 1] = offset[T + edge_count] + K1 * K2;
		}

		cerr << "T=" << T << ", n=" << n << ", nf=" << nf << ", m=" << m << ", me=" << me << endl;

		assert(row == m + me);

		ConstrInv* At = new ConstrInv[n+nf];
		transpose(A, m+me, n+nf, At);
		ins_pred_prob->A = A;
		ins_pred_prob->At = At;
		ins_pred_prob->b = b;

		//compute cost vector
		memset(c, 0.0, sizeof(double)*(n+nf));

		//unigram
		for (int i = 0; i < T; i++){
			int K = ins->node_label_lists[i]->size();
			Float* ci = ins->node_score_vecs[i];
			for (int k = 0; k < K; k++){
				c[offset[i]+k] = (double)ci[k];
			}
		}
		//bigram
		edge_count = 0;
		for (vector<pair<Int, Int>>::iterator e = ins->edges.begin(); e != ins->edges.end(); e++, edge_count++){
			int i = e->first, j = e->second;
			int K1 = ins->node_label_lists[i]->size();
			int K2 = ins->node_label_lists[j]->size();
			Float* cij = ins->edge_score_vecs[edge_count]->c;
			for (int k1k2 = 0; k1k2 < K1*K2; k1k2++){
				c[offset[T+edge_count] + k1k2] = (double)cij[k1k2];
			}
		}

		ins_pred_prob->c = c;

		ins_pred_prob->x = new double[n+nf];
		ins_pred_prob->y = new double[m+me];
		return ins_pred_prob;
	}

	static double score = 0.0;
	double LPpredict(Instance* ins, Param* param){
		//construct prediction problem for this instance
		LP_Problem* ins_pred_prob = construct_LP(ins, param);

		ins_pred_prob->solve();

		double hit = 0.0;
		int T = 0;
		if (param->problem_type == "multilabel"){ 
			//Rounding
			double* x = ins_pred_prob->x;
			int K = ins->node_label_lists[0]->size();
			T = K;
			cerr << "x:\t";
			for(int i=0;i<K;i++){
				cerr << x[i] << ":";
				double true_yk = (find(ins->labels.begin(), ins->labels.end(), i) != ins->labels.end())?1.0:0.0;
				cerr << true_yk << " ";
				hit += 1 - fabs(true_yk - x[i]);
			}
			for (int i = 0; i < ins_pred_prob->n + ins_pred_prob->nf; i++)
				score += x[i]*ins_pred_prob->c[i];
			cerr << ", score=" << score;
			cerr << endl;
		} else {
			//Rounding
			T = ins->T;
			double* x = ins_pred_prob->x;
			int offset = 0;
			//cerr << endl;
			for (int i = 0; i < T; i++){
				int true_label = ins->labels[i];
				int K = ins->node_label_lists[i]->size();
				//offset+0 to offset+K-1 is a prob distribution
				hit += x[offset+true_label];
				//cerr << "true_label=" << true_label << " ";
				for (int k = 0; k < K; k++){
					if (fabs(x[offset+k])>1e-3){
						//cerr << k << ":" << x[offset+k] << " ";
					}
				}
				//cerr << endl;
				offset += K;
			}
			for (int e = 0; e < ins->edges.size(); e++){
				int i = ins->edges[e].first, j = ins->edges[e].second;
				int K1 = ins->node_label_lists[i]->size();
				int K2 = ins->node_label_lists[j]->size();
				for (int k1 = 0; k1 < K1; k1++){
					for (int k2 = 0; k2 < K2; k2++){
						int k1k2 = k1*K2 + k2;
						if (fabs(x[offset + k1k2]) > 1e-3){
							//cerr << "(" << k1 << "," << k2 << "):" << x[offset + k1k2] << " ";
						}
					}
				}
				//cerr << endl;
			}
		}

		return hit/T;
	}

	double compute_acc_sparseLP(Problem* prob){
		vector<Instance*>* data = &(prob->data);
		double N = 0.0;
		double hit = 0.0;
		for (int n = 0; n < data->size(); n++){
			//if (n >= 10) continue;
			cerr << "@" << n << ": ";
			Instance* ins = data->at(n);
			N += ins->T;
			double acc = LPpredict(ins, prob->param);
			double temp_hit = hit;
			hit += acc*ins->T;

			cerr << ", acc=" << ((hit-temp_hit)/ins->T) << endl;
		}

		return hit/N;
	}


#endif
