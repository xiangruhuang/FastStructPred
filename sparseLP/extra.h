#ifndef EXTRA_H
#define EXTRA_H
class Seq{
	public:
	vector<SparseVec*> features;
	Labels labels;
       	Int T;
};

class ChainProblem{
	
	public:
	static map<string,Int> label_index_map;
	static vector<string> label_name_list;
	static Int D;
	static Int K;
	static Int* remap_indices;
	static Int* rev_remap_indices;
	vector<Seq*> data;
	Int N;

	ChainProblem(char* fname){
		readData(fname);
	}
	
	void readData(char* fname){
		ifstream fin(fname);
		char* line = new char[LINE_LEN];
		
		Seq* seq = new Seq();
		Int d = -1;
		N = 0;
		while( !fin.eof() ){

			fin.getline(line, LINE_LEN);
			string line_str(line);
			
			if( line_str.length() < 2 && fin.eof() ){
				if(seq->labels.size()>0)
					data.push_back(seq);
				break;
			}else if( line_str.length() < 2 ){
				data.push_back(seq);
				seq = new Seq();
				continue;
			}
			vector<string> tokens = split(line_str, " ");
			//get label index
			Int lab_ind;
			map<string,Int>::iterator it;
			if(  (it=label_index_map.find(tokens[0])) == label_index_map.end() ){
				lab_ind = label_index_map.size();
				label_index_map.insert(make_pair(tokens[0],lab_ind));
			}else
				lab_ind = it->second;

			SparseVec* x = new SparseVec();
			for(Int i=1;i<tokens.size();i++){
				vector<string> kv = split(tokens[i],":");
				Int ind = atoi(kv[0].c_str());
				Float val = atof(kv[1].c_str());
				x->push_back(make_pair(ind,val));

				if( ind > d )
					d = ind;
			}
			seq->features.push_back(x);
			seq->labels.push_back(lab_ind);
			N++;
		}
		fin.close();
		
		d += 1; //bias
		if( D < d )
			D = d;

		for(Int i=0;i<data.size();i++)
			data[i]->T = data[i]->labels.size();

		label_name_list.resize(label_index_map.size());
		
		for(map<string,Int>::iterator it=label_index_map.begin(); it!=label_index_map.end(); it++){
			label_name_list[it->second] = it->first;
		}
		
		K = label_index_map.size();
		
		delete[] line;
	}
	void label_random_remap(){
		srand(time(NULL));
		if (remap_indices == NULL || rev_remap_indices == NULL){
			remap_indices = new Int[K];
			for (Int k = 0; k < K; k++)
				remap_indices[k] = k;
			random_shuffle(remap_indices, remap_indices+K);
			rev_remap_indices = new Int[K];
			for (Int k = 0; k < K; k++)
				rev_remap_indices[remap_indices[k]] = k;
			label_index_map.clear();
			for (Int ind = 0; ind < K; ind++){
				label_index_map.insert(make_pair(label_name_list[ind], remap_indices[ind]));
			}
			label_name_list.clear();
			label_name_list.resize(label_index_map.size());
			for(map<string,Int>::iterator it=label_index_map.begin(); it!=label_index_map.end(); it++){
				label_name_list[it->second] = it->first;
			}
		}
		for (int i = 0; i < data.size(); i++){
			Seq* seq = data[i];
			for (int t = 0; t < seq->T; t++){
				Int yi = seq->labels[t];
				seq->labels[t] = remap_indices[yi];
			}
		}
	}

	void print_freq(){
		Int* freq = new Int[K];
		memset(freq, 0, sizeof(Int)*K);
		for (int i = 0; i < data.size(); i++){
			Seq* seq = data[i];
			for (int t = 0; t < seq->T; t++){
				Int yi = seq->labels[t];
				freq[yi]++;
			}
		}
		sort(freq, freq + K, greater<Int>());
		for (int k = 0; k < K; k++)
			cout << freq[k] << " ";
		cout << endl;
	}

	void simple_shuffle(){
		
		for (int i = 0; i < data.size(); i++){
			Seq* seq = data[i];
			for (int t = 0; t < seq->T; t++){
				Int yi = seq->labels[t];
				seq->labels[t] = (yi+13)%K;
			}
		}
	}
};

map<string,Int> ChainProblem::label_index_map;
vector<string> ChainProblem::label_name_list;
Int ChainProblem::D = -1;
Int ChainProblem::K;
Int* ChainProblem::remap_indices=NULL;
Int* ChainProblem::rev_remap_indices=NULL;

#endif
