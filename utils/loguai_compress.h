#ifndef LOGUAI_COMPRESS
#define LOGUAI_COMPRESS

class Var{
	public:
		int id;
		int size;
		Var(int _id, int _size){
			id = _id;
			size = _size;
		}
};

class Function{
	public:
		int size;
		Float* score_vec;
		Function(int _size){
			size = _size;
			score_vec = new Float[size];
			memset(score_vec, 0.0, sizeof(Float)*(this->size));
		}
		~Function(){
			delete score_vec;
		}
		bool isequalto(Function* f){
			Float* a = f->score_vec;
			Float* b = this->score_vec;
			int sa = f->size, sb = this->size;
			if (sa != sb)
				return false;
			for (int i = 0; i < sa; i++){
				if (a[i] != b[i]){
					return false;
				}
			}
			return true;
		}
		void print(){
			for (int i = 0; i < this->size; i++){
				if (i != 0){
					cerr << " ";
				}
				cerr << score_vec[i];
			}
			cerr << endl;
		}
};

class Factor{
	public:
		vector<Var> vars;
	 	Function* function;
		int size(){
			int s = 1;
			for (int i = 0; i < vars.size(); i++){
				s *= vars[i].size;
			}
			return s;
		}
		Factor(){}
		Factor(Var v){
			vars.clear();
			vars.push_back(v);
			function = new Function(this->size());	
		}
		Factor(Var vi, Var vj){
			vars.clear();
			vars.push_back(vi);
			vars.push_back(vj);
			function = new Function(this->size());	
		}
		~Factor(){
		}
		bool equalto(Factor& fac){
			vector<Var>& a = fac.vars;
			vector<Var>& b = this->vars;
			if (a.size() != b.size())
				return false;
			for (int i = 0; i < a.size(); i++){
				if (a[i].id != b[i].id){
					return false;
				}
			}
			return true;
		}
		
		bool has_same_score_vec(Factor* f){
			return this->function->isequalto(f->function);
		}

		void print(){
			cerr << this << "\t" << "\t";
			cerr << "factor{";
			for (int i = 0; i < vars.size(); i++){
				if (i != 0){
					cerr << ",";
				}
				cerr << vars[i].id;
			}
			cerr << "}:\t";
			function->print();
		}
};
#endif
