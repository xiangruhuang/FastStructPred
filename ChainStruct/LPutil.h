#ifndef LPUTIL_H
#define LPUTIL_H
#include <cmath>
#include <vector>
#include <queue>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <omp.h>

using namespace std;

typedef vector<pair<int,double> > Constr;
typedef vector<pair<int,double> > ConstrInv;

typedef vector<pair<int,double> > PairList;

class Tuple{
	public:
	int first;
	int second;
	double third;
	Tuple(int f, int s, double t):first(f),second(s),third(t){}
};
typedef vector<Tuple> TupleList;

void writeVec(char* fname, double* v, int n){
	
	 ofstream fout(fname);
	 for(int j=0;j<n;j++){
		if( fabs(v[j]) > 1e-2 )
			fout << j+1 << "\t" << v[j] << endl; 
	 }
	 fout.close();
}

void readMeta(const char* fname, int& n, int& nf, int& m, int& meq){
	
	ifstream fin(fname);
	char* tmp = new char[LINE_LEN];
	fin >> tmp >> n;
	fin >> tmp >> nf;
	fin >> tmp >> m;
	fin >> tmp >> meq;

	fin.close();

	delete[] tmp;
}

/** read "size" by 1 vector in fname into v[offset:end] 
 */
void readVec(const char* fname, double* v, int offset, int size){
	
	double val ;
	ifstream fin(fname);
	for(int i=offset; i<offset+size; i++){	
		fin >> val;
		v[i] = val;
	}
	fin.close();
}

void readMat(const char* fname, int m, int n, Constr* A, int row_offset){
	
	int i,j;
	double val;
	ifstream fin(fname);
	fin >> i >> j >> val; //filter one line
	if( i != m || j != n ){
		cerr << "dimension in " << fname << " does not match that in meta file" << endl;
		exit(0);
	}
	
	while( !fin.eof() ){
		
		fin >> i >> j >> val;
		if( fin.eof() )
			break;

		if( i-1 >= m || j-1 >= n ){
			cerr << "index:" << "(" << i-1 << ", " << j-1 << ") out of bound when reading " << fname << endl; 
			exit(0);
		}
		A[row_offset+i-1].push_back(make_pair(j-1,val));
	}
	fin.close();
}

void transpose(Constr* A, int m, int n, ConstrInv* At){
	
	for(int i=0;i<m;i++){
		
		for(Constr::iterator it=A[i].begin(); it!=A[i].end(); it++){
			
			At[it->first].push_back(make_pair(i,it->second));
		}
	}
}

double objective(int n, int m, int meq, double* x, double* c, double* b, double* beq, ConstrInv* At, ConstrInv* Aeqt, 
		double* alpha_t, double* beta_t, double eta_t){
	
	double fval = 0.0;
	for(int j=0;j<n;j++)
		fval += c[j] * x[j];

	double* w = new double[m];
	double* v = new double[meq];
	for(int i=0;i<m;i++)
		w[i] = -b[i] + alpha_t[i]/eta_t;
	for(int i=0;i<meq;i++)
		v[i] = -beq[i] + beta_t[i]/eta_t;

	double tmp;
	for(int j=0;j<n;j++){
		tmp = x[j];
		for(ConstrInv::iterator it=At[j].begin();it!=At[j].end();it++)
			w[it->first] += tmp*it->second;
		for(ConstrInv::iterator it=Aeqt[j].begin();it!=Aeqt[j].end();it++)
			v[it->first] += tmp*it->second;
	}
	
	tmp = 0.0;
	for(int i=0;i<m;i++){
		if( w[i] > 0.0 )
			tmp += w[i]*w[i];
	}
	fval += eta_t*tmp / 2.0;

	tmp = 0.0;
	for(int i=0;i<meq;i++){
		tmp += v[i]*v[i];
	}
	fval += eta_t*tmp / 2.0;

	return fval;
}

double dual_inf(int n, int nf, int m, int me, double* w, ConstrInv* At, double* c, double eta_t){

	double max_val = 0.0;
	//find ||(-eta*A'w-c)_+||_{\infty}
	for(int j=0;j<n;j++){
		double Atv_c = 0.0;
		for(ConstrInv::iterator it=At[j].begin(); it!=At[j].end(); it++){
			if( it->first >= m || w[it->first] > 0 )
				Atv_c -= it->second*w[it->first];
		}
		Atv_c *= eta_t;
		Atv_c -= c[j];
		
		if( Atv_c > max_val )
			max_val = Atv_c;
	}
	//find ||eta*A'w+cf||_{\infty}
	for(int j=n;j<nf;j++){
		double Atv_cf = 0.0;
		for(ConstrInv::iterator it=At[j].begin(); it!=At[j].end(); it++){
			if( it->first >= m || w[it->first] > 0 )
				Atv_cf += it->second*w[it->first];
		}
		Atv_cf *= eta_t;
		Atv_cf += c[j];
		double tmp = fabs(Atv_cf);
		if( tmp > max_val )
			max_val = tmp;
	}

	return max_val;
}

double primal_inf(int n, int nf, int m, int me, double* x, Constr* A, double* b){
	
	double max_val = 0.0;
	//find ||(Ax-b)_+||_{\infty}
	for(int i=0;i<m;i++){
		double Ax_b = -b[i];
		for(Constr::iterator it=A[i].begin(); it!=A[i].end(); it++){
			Ax_b += it->second*x[it->first];
		}
		if( Ax_b > max_val )
			max_val = Ax_b;
	}
	//find ||Ae*x-b||_{\infty}
	for(int i=m;i<m+me;i++){
		double Aex_b = -b[i];
		for(Constr::iterator it=A[i].begin(); it!=A[i].end(); it++){
			Aex_b += it->second*x[it->first];
		}
		double tmp = fabs(Aex_b);
		if( tmp > max_val )
			max_val = tmp;
	}
	
	return max_val;
	//return sum/(m+me);
}


double duality_gap(int n, int nf, int m, int me, double* x, double* w, double* c, double* b, double eta_t){
	
	double cx = 0.0;
	for(int j=0;j<n+nf;j++)
		cx += c[j]*x[j];
	double bw = 0.0;
	for(int i=0;i<m;i++){
		if(w[i] > 0.0)
			bw += b[i]*w[i];
	}
	for(int i=m;i<m+me;i++)
		bw += b[i]*w[i];

	return  fabs(cx + bw*eta_t) ;
}

void sort_and_unique( vector<int>& index ){

	sort( index.begin(), index.end() );
	index.erase( unique( index.begin(), index.end()), index.end() );
}


void add_relev_index( vector<int>& active_var_index, ConstrInv* A, vector<int>& rel_index){
	
	for(int r=0;r<active_var_index.size();r++){
		int j = active_var_index[r];
		for(ConstrInv::iterator it=A[j].begin(); it!=A[j].end(); it++){
			rel_index.push_back(it->first);
		}
	}
	sort_and_unique( rel_index );
}

void find_relev_index( vector<int>& active_var_index, ConstrInv* A, vector<int>& rel_index){
	
	rel_index.clear();
	add_relev_index(active_var_index, A, rel_index);
}

void negate_mat(PairList* mat, int nrows){
	
	for(int i=0;i<nrows;i++){
		for(PairList::iterator it=mat[i].begin();it!=mat[i].end();it++){
			it->second *= -1.0;
		}
	}
}

#endif
