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
#include "CG.h"

using namespace std;

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
		
		tol = 1e-2;
		//tol_trans = 10*tol;
		tol_trans = -1;
		tol_sub = 0.5*tol_trans;
		
		eta = 1.0;
		nnz_tol = 1e-4;
		max_iter = 1000;
	}
};	

LP_Param param;

/** Using Randomized Coordinate Descent to solve:
 *
 *  min c'x + \frac{eta_t}{2}\| (Ax-b+alpha_t/eta_t)_+ \|^2 + \frac{eta_t}{2}\|Aeq*x-t*beq+beta_t/eta_t\|^2
 *  s.t. x >= 0
 */
void rcd(int n, int nf, int m, int me, ConstrInv* At, double* b, double* c, double* x, double* w, double* h2_jj, double* hjj_ubound, double eta_t, int& niter, int inner_max_iter, int& active_matrix_size, double& PGmax_old_last, int phase){
	
	int max_num_linesearch = 20;
	double sigma = 0.01; //for line search, find smallest t \in {0,1,2,3...} s.t.
			     // F(x+beta^t*d)-F(x) <= sigma * beta^t g_j*d
	
	//initialize active index
	int* index = new int[n+nf];
	int active_size = n+nf;
	for(int j=0;j<active_size;j++)
		index[j] = j;
	
	int iter=0;
	double PG, PGmax_old = 1e300, PGmax_new;
	double d;
	while(iter < inner_max_iter){
		
		PGmax_new = -1e300;
		random_shuffle(index, index+active_size);
		
		for(int s=0;s<active_size;s++){
			
			int j = index[s];
			//cerr << "j=" << j << endl;
			
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
		}
		iter++;
		if( iter % 10 == 0 ){
			//cerr << "." ;
		}
		
		PGmax_old = PGmax_new;
		if( PGmax_old <= 0.0 )
			PGmax_old = 1e300;
		
		if( PGmax_new <= param.tol_sub ){ //reach stopping criteria
			
			//cerr << "*" ;
			break;
		}
	}
	//passing some info
	niter = iter;
	active_matrix_size = 0;
	for(int s=0;s<active_size;s++)
		active_matrix_size += At[index[s]].size();
	PGmax_old_last = PGmax_old;
	
	delete[] index;
}

/** Solving Linear System:
 *  H*delta_x = -g,
 *  where H = eta_t * ( A_sub'*A_sub + Aeq_sub'*Aeq_sub )
 *  	  g = c_sub + eta_t * ( A_sub'*w + Aeq_sub'*v )
 *
 *  	  w = [A*x_sub - b + w_{t-1}]_+
 *  	  v = Aeq*x_sub - beq + v_{t-1}
 */
class LPAugFunc : public Function {
	
	public:
	LPAugFunc(double _eta, int _n, int _m, int _meq, ConstrInv* _At){ 
		n = _n;
		m = _m;
		meq = _meq;
		At = _At;
		eta = _eta;
	}
	
	virtual void Hv(double* s, double* Hs){
		
		for(int j=0;j<n;j++)
			Hs[j] = 0.0;
		
		//w = [DA*s ; Aeq*s]
		double* w = new double[m+meq];
		for(int i=0;i<m+meq;i++)
			w[i] = 0.0;
		for(int j=0;j<n;j++){
			double tmp = s[j];
			for(ConstrInv::iterator it=At[j].begin(); it!=At[j].end(); it++)
				w[it->first] += it->second*tmp;
		}
		//Hs = [A' Aeq'] * w
		for(int j=0;j<n;j++){
			double sum = 0.0;
			for(ConstrInv::iterator it=At[j].begin(); it!=At[j].end(); it++)
				sum += w[it->first]*it->second;
			Hs[j] += sum;
		}
		delete[] w;
		
		//eta_t*(....)
		for(int j=0;j<n;j++){
			Hs[j] *= eta;
			Hs[j] += 1e-2*s[j]; //for case H=0
		}
	}

	virtual int get_nr_variable(){
		return n;
	}
	
	
	double eta;
	ConstrInv* At;
	int m;
	int meq;
	int n;
};
void solve_linear_system(int n, int m, int me, double eta, ConstrInv* At_sub, double* g_sub, double* delta_x, double prec){
	
	int max_iter = n;
	
	Function* fun = new LPAugFunc( eta, n, m, me, At_sub );
	CG* cgsolve = new CG(fun, prec, max_iter);
	
	//initialized as 0 to ensure x in the column space of H
	double* r = new double[n];
	int cg_iter = cgsolve->cg(g_sub, delta_x, r);
	//cerr << "#CG=" << cg_iter << ", ";
	
	delete[] r;
}

/** Using Projected Newton Method to solve:
 *
 *  min c'x + \frac{eta_t}{2}\| (Ax-b+alpha_t/eta_t)_+ \|^2 + \frac{eta_t}{2}\|Aeq*x-t*beq+beta_t/eta_t\|^2
 *  s.t. x >= 0
 */

void projected_newton(int n, int nf, int m, int me, Constr* A, ConstrInv* At, double* b, double* c, double*& x, double* w, double eta_t, int& niter, int inner_max_iter){
	
	double sigma = 0.01;  //line search parameter
	double beta = 0.5;
	double cg_prec = 0.1;
	int max_line_search = 100;
	vector<int> active_index;
	vector<int> rel_A_index;
	ConstrInv* At_sub = new ConstrInv[n+nf];
	double* g = new double[n+nf];
	double prec = param.tol_sub;
	
	double last_PG_max = 1e300;
	niter = 0;
	while( niter < inner_max_iter ){
		
		//1. Compute gradient
		for(int j=0;j<n+nf;j++)
			g[j] = 0.0;
		for(int i=0;i<m;i++){
			double tmp = w[i];
			for(Constr::iterator it=A[i].begin(); it!=A[i].end(); it++)
				if( tmp > 0.0 )
					g[it->first] += tmp * it->second;
		}
		for(int i=m;i<m+me;i++){
			double tmp = w[i];
			for(Constr::iterator it=A[i].begin(); it!=A[i].end(); it++)
				g[it->first] += tmp * it->second;
		}
		for(int j=0;j<n+nf;j++)
			g[j] = eta_t*g[j] + c[j];
		////Projected Gradient for stopping condition
		double PG_max = -1e300;
		for(int j=0;j<n;j++){
			if( x[j] > 0.0 )
				PG_max = max( PG_max, fabs(g[j]) );
			else{
				if( g[j] < 0.0 )
					PG_max = max( PG_max, -g[j] );
			}
		}
		for(int j=n;j<n+nf;j++)
			PG_max = max( PG_max, fabs(g[j]) );
		
		if(niter % 10==0){
			//cerr << "-" ;
			if( PG_max < last_PG_max ){
				//cerr << "|PG|=" << PG_max ;
				last_PG_max = PG_max;
			}
		}
		if( PG_max < prec )
			break;
		
		//2. Finding indexes of nonzeros in x
		active_index.clear();
		for(int j=0;j<n;j++){
			if( x[j] > 0.0 || g[j] < 0.0 )
				active_index.push_back(j);
		}
		for(int j=n;j<n+nf;j++)
			active_index.push_back(j);
		
		int active_size = active_index.size();
		//3. Construct At_sub, Aeqt_sub, g_sub
		for(int r=0;r<active_size;r++){
			int j = active_index[r];
			At_sub[r].clear();
			for(ConstrInv::iterator it=At[j].begin();it!=At[j].end();it++){
				if( it->first >= m || w[it->first] > 0.0 ){ //is binding constraint
					At_sub[r].push_back(make_pair(it->first, it->second));
				}
			}
		}
		double* g_sub = new double[active_size];
		for(int r=0;r<active_size;r++){
			int j = active_index[r];
			g_sub[r] = g[j];
		}
		
		//4. Solving a Linear System
		double* delta_x_sub = new double[active_size];
		solve_linear_system(active_size, m, me, eta_t, At_sub, g_sub, delta_x_sub, cg_prec);
		
		//5. Identify relevant constraints w.r.t. active_set
	       	find_relev_index(active_index, At, rel_A_index);
		
		//5. Line search (finds alpha)
		double alpha = 1.0, alpha_old = 0.0, cond;
		////compute old Function value
		double fval_old = 0.0;
		for(vector<int>::iterator it=rel_A_index.begin(); it!=rel_A_index.end(); it++){
			double tmp = w[*it];
			if( *it >= m || tmp > 0.0 )
				fval_old += tmp*tmp;
		}
		fval_old *= eta_t/2;
		for(int r=0;r<active_index.size();r++){
			int j = active_index[r];
			fval_old += c[j]*x[j];
		}
		
		////main line-search loop
		double* x_sub_old = new double[active_size];
		for(int r=0;r<active_size;r++)
			x_sub_old[r] = x[active_index[r]];
		int t;
		for(t=0;t<max_line_search;t++){
			////compute gradient decrement
			double grad_decre = 0.0;
			for(int r=0;r<active_size;r++){
				int j = active_index[r];
				double x_old = x[j];
				double x_new = alpha*delta_x_sub[r] + x_old;
				if( j<n & x_new < 0 )
					x_new = 0.0;
				grad_decre += g_sub[r]*(x_new-x_old) ;
			}
			cond = -sigma * grad_decre;
			////update w, v to reflect new value (alpha*delta_x + x_old)
			for(int r=0;r<active_size;r++){
				int j = active_index[r];
				double x_new = alpha*delta_x_sub[r] + x[j];
				if( j<n && x_new < 0 )
					x_new = 0.0;
				
				for(ConstrInv::iterator it=At[j].begin(); it!=At[j].end(); it++)
					w[it->first] += it->second*(x_new-x_sub_old[r]);
				
				x_sub_old[r] = x_new;
			}
			////compute new Function value
			double fval_new = 0.0;
			for(vector<int>::iterator it=rel_A_index.begin(); it!=rel_A_index.end(); it++){
				double tmp = w[*it];
				if( *it >= m || tmp > 0.0 )
					fval_new += tmp*tmp;
			}
			fval_new *= eta_t/2;
			for(int r=0;r<active_size;r++){
				int j = active_index[r];
				double x_new = alpha*delta_x_sub[r] + x[j];
				if( j<n && x_new < 0 )
					x_new = 0.0;
				fval_new += c[j]*x_new;
			}
			cond = cond + fval_new - fval_old;
			////Test Line-Search Condition
			if( cond <= 0.0 )
				break;
			else
				alpha = alpha * beta;
		}
		//cerr << "alpha=" << alpha << endl;
		if(  t==max_line_search ){
			//cerr << "reach maximum #linesearch" << endl;
			return ;
			//continue;
		}
		
		//6. Update x
		for(int r=0;r<active_size;r++){
			int j = active_index[r];
			double x_new = alpha*delta_x_sub[r] + x[j];
			if( j < n && x_new < 0.0 )
				x_new = 0.0;
			
			x[j] = x_new;
		}
		
		
		delete[] g_sub;
		delete[] delta_x_sub;
		delete[] x_sub_old;
		niter++;
	}
	
	delete[] At_sub;
	delete[] g;
}

/** Solve:
 *
 *  min  c'x
 *  s.t. Ax <= b
 *       Aeq x = beq
 *       x >= 0
 */
void LPsolve(int n, int nf, int m, int me, Constr* A, ConstrInv* At, double* b, double* c, double*& x, double*& w){
	
	double eta_t = param.eta;
	int max_iter = param.max_iter;
	
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
	int print_per_iter = 2;
	double pinf = 1e300, gap=1e300, dinf=1300, obj; //primal infeasibility
	int nnz;//, nnz_last=1;
	int active_nnz = nnz_A;
	int phase = 1, inner_iter;
	double dinf_last=1e300, pinf_last=1e300, gap_last=1e300;
	for(int t=0;t<max_iter;t++){
		
		if( phase == 1 ){
			inner_max_iter = (active_nnz!=0)?(nnz_A/active_nnz):(n+nf);
			//inner_max_iter = 1;
			rcd(n,nf,m,me, At,b,c,  x, w, h2_jj, hjj_ubound, eta_t, niter, inner_max_iter, active_nnz, PGmax_old, phase);
			
		}else if( phase == 2 ){
			
			if( !param.use_CG ) rcd(n,nf,m,me, At,b,c,  x, w, h2_jj, hjj_ubound, eta_t, niter, inner_max_iter, active_nnz, PGmax_old, phase);
			else	      projected_newton(n,nf,m,me, A,At,b,c,  x, w, eta_t, niter, inner_max_iter);
			
			//cerr << endl;
		}
		//niter = inner_iter;
		
		if( t % print_per_iter ==0 ){

			nnz = 0;
			for(int j=0;j<n+nf;j++)
				if( x[j] > param.nnz_tol )
					nnz++;
			
			obj = 0.0;
			for(int j=0;j<n+nf;j++)
				obj += c[j]*x[j];
			
			pinf = primal_inf(n,nf,m,me,x,A,b);
			dinf = dual_inf(n,nf,m,me,w,At,c,eta_t);
			gap = duality_gap(n,nf,m,me,x,w,c,b,eta_t);

			
			//cerr << setprecision(7) << "iter=" << t << ", #inner=" << niter << ", obj=" << obj ;
			//cerr << setprecision(2)  << ", p_inf=" << pinf << ", d_inf=" << dinf << ", gap=" << fabs(gap/obj) << ", nnz=" << nnz << "(" << ((double)active_nnz/nnz_A) << ")" ;
			//cerr << endl;
		}
		
		if( pinf<=param.tol && dinf<=param.tol ){
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
		
		if( phase == 1 && pinf <= param.tol_trans ){
			
			phase = 2;
			print_per_iter = 1;
			//cerr << "phase = 2" << endl;
			inner_max_iter = 100;
		}
		
		if( phase == 2 ){
			
			if( (niter < 2) || (pinf < 0.5*dinf && dinf>dinf_last) ){
				
				if(niter < inner_max_iter) param.tol_sub *= 0.5;
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
	param.eta = eta_t;
}


#endif
