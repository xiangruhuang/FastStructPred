#include "util.h"
#include "chain.h"
#include <cassert>

extern long long line_bottom, line_top;
extern long long mat_bottom, mat_top;
long long submat_bottom, submat_top;
extern double overall_time;

class BCFWsolve{
	
	public:
	enum Direction {F_LEFT=0, F_RIGHT=1, NUM_DIRECT};
	
	BCFWsolve(Param* param){
		
		//Parse info from ChainProblem
		par = param;
		using_brute_force = param->using_brute_force;
		do_subSolve = param->do_subSolve;
		heldout_period = param->heldout_period;
		split_up_rate = param->split_up_rate;
		prob = param->prob;
		heldout_prob = param->heldout_prob;
		data = &(prob->data);
		nSeq = data->size();
		D = prob->D;
		K = prob->K;
		modelFname = param->modelFname;	
		C = param->C;
		eta = param->eta;
		max_iter = param->max_iter;
		write_model_period = param->write_model_period;
		early_terminate = param->early_terminate;
		admm_step_size = param->admm_step_size; //1.0/split_up_rate;
		interval_start = new Int[split_up_rate];
		interval_end = new Int[split_up_rate];
		Int length = K / split_up_rate;
		for (Int i = 0; i < split_up_rate; i++){
			interval_start[i] = i*length;
			interval_end[i] = (i+1)*length;
		}
		interval_end[split_up_rate-1] = K;

		//bi_search
		inside = new bool[K*K];
		memset(inside, 0, sizeof(bool)*K*K);
		col_heap = new ArrayHeap[K];
		row_heap = new ArrayHeap[K];
		for (int k = 0; k < K; k++){
			row_heap[k] = new pair<Float, Int>[K];
			col_heap[k] = new pair<Float, Int>[K];	
		}	
		v_heap = new pair<Float, Int>[K*K];
		col_heap_size = new Int[K];
		row_heap_size = new Int[K];
		col_index = new Int[K*K];
		row_index = new Int[K*K];
		v_index = new Int[K*K];
		col_dir = new Int[K*K];
		row_dir = new Int[K*K];
		for (int kk = 0; kk < K*K; kk++){
			row_dir[kk] = kk / K;
			col_dir[kk] = kk % K;
		}
		
		//Compute unigram and bigram offset[i] = \sum_{j=1}^{i-1} T_j
		compute_offset();
		N = uni_offset[nSeq-1] + data->at(nSeq-1)->T; //#unigram factor
		M = bi_offset[nSeq-1] + data->at(nSeq-1)->T - 1; //#bigram factor
		//allocate dual variables
		/*alpha = new Float*[N];
		for(Int i=0;i<N;i++){
			alpha[i] = new Float[K];
			for(Int k=0;k<K;k++)
				alpha[i][k] = 0.0;
		}
		beta = new Float*[M];
		for(Int i=0;i<M;i++){
			beta[i] = new Float[K*K];
			for(Int kk=0;kk<K*K;kk++)
				beta[i][kk] = 0.0;
		}*/
		
		//maintain message
		/*beta_suml = new Float*[M];
		for(Int i=0;i<M;i++){
			beta_suml[i] = new Float[K];
			for(Int k=0;k<K;k++)
				beta_suml[i][k] = 0.0;
		}
		beta_sumr = new Float*[M];
		for(Int i=0;i<M;i++){
			beta_sumr[i] = new Float[K];
			for(Int k=0;k<K;k++)
				beta_sumr[i][k] = 0.0;
		}*/
		msg_left  = new Float*[M];
		msg_right = new Float*[M];
		pos_left.clear(); pos_right.clear();
		for (Int i = 0; i < M; i++){
			msg_left[i]  = new Float[K];
			msg_right[i] = new Float[K];
			memset(msg_left[i] , 0.0, sizeof(Float)*K);
			memset(msg_right[i], 0.0, sizeof(Float)*K);
		}
		
		//allocate primal variables
		w = new Float*[D];
		for(Int j=0;j<D;j++){
			w[j] = new Float[K];
			for(Int k=0;k<K;k++)
				w[j][k] = 0.0;
		}
		v = new Float*[K];
		for(Int k=0;k<K;k++){
			v[k] = new Float[K];
			for(Int k2=0;k2<K;k2++){
				v[k][k2] = 0.0;
				//update_v_heap
				v_heap[k*K+k2] = make_pair(0.0, k*K+k2);
				v_index[k*K+k2] = k*K+k2;
				//update_row_heap[k]
				row_heap[k][k2] = make_pair(0.0, k*K+k2);
				row_index[k*K+k2] = k2;
				//update_col_heap[k2]
				col_heap[k2][k] = make_pair(0.0, k*K+k2);
				col_index[k*K+k2] = k;
			}
			row_heap_size[k] = K;
			col_heap_size[k] = K;
		}
		model = new Model(w, v, prob);
		v_heap_size = K*K;
		
		//allocating Lagrangian Multipliers for consistency constraInts
		/*mu = new Float*[2*M]; //2 because of bigram
		//messages = new Float*[2*M];
		for(Int i=0;i<2*M;i++){
			mu[i] = new Float[K];
			//messages[i] = new Float[K];
			for(Int k=0;k<K;k++)
				mu[i][k] = 0.0;
			//for(Int k=0;k<K;k++)
			//	messages[i][k] = 0.0;
		}*/
		
		//pre-allocate some algorithmic constants
		Q_diag = new Float[N];
		for(Int n=0;n<nSeq;n++){
			Seq* seq = data->at(n);
			for(Int t=0;t<seq->T;t++){
				Int i = uni_index( n, t );
				Q_diag[i] = 0.0;
				for(SparseVec::iterator it=seq->features[t]->begin(); it!=seq->features[t]->end(); it++){
					Q_diag[i] += it->second * it->second;
				}
			}
		}

		//uni_search
		max_indices = new Int[K];
		w_nz_index = new vector<Int>[D];
		
		//global buffer
		prod = new Float[K];
		grad = new Float[K*K];
		grad_heap = new Float[K*K];
		zero_msg = new Float[K];
		memset(zero_msg, 0.0, sizeof(Float)*K);
	}
	
	~BCFWsolve(){
		
		delete[] uni_offset;
		delete[] bi_offset;
		//primal variables
		for(Int j=0;j<D;j++)
			delete[] w[j];
		delete[] w;
		for(Int k=0;k<K;k++)
			delete[] v[k];
		delete[] v;
	
		//maintain message	
		for (Int i=0; i<M; i++){
			delete[] msg_left[i];
			delete[] msg_right[i];
		}
		pos_left.clear(); pos_right.clear();
		delete[] msg_left;
		delete[] msg_right;
		

		//some constants
		delete Q_diag;

		//uni_search
		delete[] max_indices;
		delete[] w_nz_index;
		delete[] interval_start;
		delete[] interval_end;

		//global buffer
		delete[] grad;
		delete[] prod;
		delete[] grad_heap;
		delete[] zero_msg;

		//bi_search
		delete[] inside;
		for (int k = 0; k < K; k++){
			delete[] col_heap[k];
			delete[] row_heap[k];
		}
		delete[] col_heap;
		delete[] row_heap;
		delete[] v_heap;
		delete[] col_heap_size;
		delete[] row_heap_size;
		delete[] col_index;
		delete[] row_index;
		delete[] v_index;
		delete[] col_dir;
		delete[] row_dir;
	}
		
	Model* solve(){
	
		srand(time(NULL));	
		
		//dump some parameters
		cerr << "split_up_rate=" << split_up_rate << ", admm_step_size=" << admm_step_size << ", C=" << C << ", eta=" << eta << endl;

		Int* uni_ind = new Int[N];
		for(Int i=0;i<N;i++)
			uni_ind[i] = i;
		Int* bi_ind = new Int[M];
		for(Int i=0;i<M;i++)
			bi_ind[i] = i;
		
		//BDMM main loop
		Float* alpha_new = new Float[K];
		Float* beta_new = new Float[K*K];
		Float p_inf;

		//BCFW
		act_k_index = new vector<pair<Int, Float>>[N];
		for (Int i = 0; i < N; i++){
			Int n,t;
			get_uni_rev_index(i, n, t);
			Seq* seq = data->at(n);
			act_k_index[i].push_back(make_pair(seq->labels[t], 0.0));
		}
		act_kk_index = new vector<pair<Int, Float>>[M];
		for (Int i = 0; i < M; i++){
			Int n,t;
			get_bi_rev_index(i, n, t);
			Seq* seq = data->at(n);
			act_kk_index[i].push_back(make_pair(seq->labels[t]*K + seq->labels[t+1], 0.0));
		}

		Float max_heldout_test_acc = -1;
		Int terminate_counting = 0;
		for(Int iter=0;iter<max_iter;iter++){

			double bi_search_time = 0.0, bi_subSolve_time = 0.0, bi_maintain_time = 0.0;
			double uni_search_time = 0.0, uni_subSolve_time = 0.0, uni_maintain_time = 0.0;
			double calcAcc_time = 0.0, admm_maintain_time = 0.0;
			area1_time = 0.0; area23_time = 0.0; area4_time = 0.0;
			line_top = 0; line_bottom = 0;
			mat_top = 0;  mat_bottom = 0;
			submat_top = 0; submat_bottom = 0;
			uni_maintain_time -= get_current_time();
			random_shuffle(uni_ind, uni_ind+N);
			uni_maintain_time += get_current_time();
			//update unigram dual variables
			for(Int r=0;r<N;r++){

				Int i = uni_ind[r];
				Int n, t;
				get_uni_rev_index(i, n, t);
				Seq* seq = data->at(n);
				Int il = -1;
				if (t != 0){
					il = bi_index(n, t-1);
				}
				Int ir = -1;
				if (t != seq->T - 1){
					ir = bi_index(n,t);
				}
				SparseVec* xi = seq->features[t];
				Int yi = seq->labels[t];
			
				//brute force search
				uni_search_time -= get_current_time();
				uni_search(i, n, t, il, ir, act_k_index[i]);
				uni_search_time += get_current_time();
				
				
				//subproblem solving
				//Float loss_per_node = 1.0/seq->T;
				Float loss_per_node = 1.0;
				uni_subSolve_time -= get_current_time();
				if (do_subSolve){
					uni_subSolve(i, n, t, il, ir, act_k_index[i], alpha_new, loss_per_node);
				} else {
					uni_update(i, n, t, il, ir, act_k_index[i], alpha_new, loss_per_node, iter);
				}
				uni_subSolve_time += get_current_time();
				//maIntain relationship between w and alpha
				uni_maintain_time -= get_current_time();
				for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
					Int j = it->first;
					Float fval = it->second;
					Float* wj = w[j];
					for(vector<pair<Int, Float>>::iterator it2 = act_k_index[i].begin(); it2 != act_k_index[i].end(); it2++){
						Int k = it2->first;
						wj[k] += fval * (alpha_new[k]-it2->second);
					}
				}
				//maintain messages(alpha) = (E*beta-alpha+\frac{1}{eta}mu)

				bool has_zero = 0;
				//update alpha
				Float* msg_right_il = NULL;
				Float* msg_left_ir  = NULL;
				if (il != -1) msg_right_il = msg_right[il];
				if (ir != -1) msg_left_ir  = msg_left[ir];
				for(vector<pair<Int, Float>>::iterator it = act_k_index[i].begin(); it != act_k_index[i].end(); it++ ){
					Int k = it->first;
					Float delta_alpha = alpha_new[k] - it->second;
					if (il != -1){
						msg_right_il[k] -= delta_alpha;
						//assert(fabs(msg_right_il[k] - mu[il*2+1][k] - beta_sumr[il][k] + alpha_new[k]) < 1e-3);
					}
					if (ir != -1){
						msg_left_ir[k]  -= delta_alpha;
						//assert(fabs(msg_left_ir[k]  - mu[ir*2][k]   - beta_suml[ir][k] + alpha_new[k]) < 1e-3);
					}
					it->second = alpha_new[k];
					has_zero |= (fabs(alpha_new[k])<=1e-12 && k != yi);
				}
					
				if (has_zero){
					vector<pair<Int, Float>> tmp_vec;
					tmp_vec.reserve(act_k_index[i].size());
					for (vector<pair<Int, Float>>::iterator it = act_k_index[i].begin(); it != act_k_index[i].end(); it++){
						Int k = it->first;
						if ( fabs(it->second) > 1e-12 || k == yi){
							tmp_vec.push_back(make_pair(k, it->second));
						}
					}
					act_k_index[i] = tmp_vec;
				}
				uni_maintain_time += get_current_time();
			}
			
			//update bigram dual variables
			random_shuffle(bi_ind, bi_ind+M);
			for(Int r=0;r<M;r++){
				Int i = bi_ind[r];
				Int n, t;
				get_bi_rev_index(i, n, t);
				Seq* seq = data->at(n);
				Int yi_l = seq->labels[t];
				Int yi_r = seq->labels[t+1];
				Int ylyr = yi_l*K + yi_r;
				Int il = uni_index(n, t);
				Int ir = uni_index(n, t + 1);
					
				//search using oracle
				bi_search_time -= get_current_time();
				if (using_brute_force){
					bi_brute_force_search(i, n, t, act_kk_index[i]);
				}else{
					bi_search(i, n, t, act_kk_index[i]);
				}
				bi_search_time += get_current_time();
				
				//subproblem solving
				bi_subSolve_time -= get_current_time();
				if (do_subSolve)
					bi_subSolve(i, n, t, act_kk_index[i], beta_new);
				else
					bi_update(i, n, t, act_kk_index[i], beta_new);
				bi_subSolve_time += get_current_time();

				//maIntain relationship between v and beta
				//maintain messages(beta) = (E*beta-alpha+\frac{1}{eta}mu)
				bi_maintain_time -= get_current_time();
				Float* msg_left_i = msg_left[i];
				Float* msg_right_i = msg_right[i];
				Int num_zero = 0;
				vector<pair<Int, Float>>::iterator to_be_trimed;
				for(vector<pair<Int, Float>>::iterator it = act_kk_index[i].begin(); it != act_kk_index[i].end(); it++){
					Int k1k2 = it->first;
					Float delta_beta = beta_new[k1k2] - it->second;
					it->second = beta_new[k1k2];
					if (fabs(it->second) < 1e-12 && it->first != ylyr ) {
						num_zero++;
						to_be_trimed = it;
					}
					if (fabs(delta_beta) < 1e-12)
						continue;
					Int k2 = k1k2 % K;
					Int k1 = k1k2 / K;
					v[k1][k2] += delta_beta;
					msg_left_i[k1] += delta_beta;
					msg_right_i[k2] += delta_beta;
					//maintain v_heap
					Int index_k1k2 = v_index[k1k2];
					v_heap[index_k1k2].first += delta_beta;
					
					//maintain row_heap
					Int row_index_k1k2 = row_index[k1k2];
					row_heap[k1][row_index_k1k2].first += delta_beta;
					
					//maintain col_heap
					Int col_index_k1k2 = col_index[k1k2];
					col_heap[k2][col_index_k1k2].first += delta_beta;
					
					if (delta_beta >= 0.0){
						siftUp(v_heap, index_k1k2, v_index);
						siftUp(row_heap[k1], row_index_k1k2, row_index);
						siftUp(col_heap[k2], col_index_k1k2, col_index);
					} else {
						siftDown(v_heap, index_k1k2, v_index, v_heap_size);
						siftDown(row_heap[k1], row_index_k1k2, row_index, row_heap_size[k1]);
						siftDown(col_heap[k2], col_index_k1k2, col_index, col_heap_size[k2]);
					}
				}

				//shrink active set if necessary
				if (num_zero > 0){
					if (num_zero > 1){
						vector<pair<Int, Float>> tmp_vec;
						tmp_vec.reserve(act_kk_index[i].size());
						for (vector<pair<Int, Float>>::iterator it = act_kk_index[i].begin(); it != act_kk_index[i].end(); it++){
							Int k1k2 = it->first;
							if ( fabs(it->second) > 1e-12 || k1k2 == ylyr){
								tmp_vec.push_back(make_pair(k1k2, it->second));
							}
						}
						//act_kk_index[i].clear();
						act_kk_index[i] = tmp_vec;
					} else {
						act_kk_index[i].erase(to_be_trimed);
					}
				} 
				bi_maintain_time += get_current_time();
			}

			//ADMM update (enforcing consistency)
			admm_maintain_time -= get_current_time();
			p_inf = 0.0;
			Float* cache = new Float[K];
			Float* infea_left = new Float[K];
			Float* infea_right = new Float[K];
			Float infea = 0.0;
			for(Int n=0;n<nSeq;n++){
				Seq* seq = data->at(n);
				for(Int t=0;t<seq->T-1;t++){
					Int i2 = bi_index(n,t);
					Int i1 = uni_index(n,t);
					Float* msg_left_i2 = msg_left[i2];
					Float* msg_right_i2 = msg_right[i2];
					memset(infea_left, 0.0, sizeof(Float)*K);
					memset(infea_right, 0.0, sizeof(Float)*K);
					for (vector<pair<Int, Float>>::iterator it = act_kk_index[i2].begin(); it != act_kk_index[i2].end(); it++){
						Int k1k2 = it->first;
						Int k1 = k1k2 / K;
						Int k2 = k1k2 % K;
						
						infea -= fabs(infea_left[k1]);
						infea_left[k1] += it->second;
						infea += fabs(infea_left[k1]);

						infea -= fabs(infea_right[k2]);
						infea_right[k2] += it->second;
						infea += fabs(infea_right[k2]);

						Float delta_mu_beta = admm_step_size*(it->second);
						msg_left_i2[k1] += delta_mu_beta;
						msg_right_i2[k2] += delta_mu_beta;
					}
					for (vector<pair<Int, Float>>::iterator it = act_k_index[i1].begin(); it != act_k_index[i1].end(); it++){
						Int k = it->first;
						Float delta_mu_alpha = admm_step_size*it->second;

						infea -= fabs(infea_left[k]);
						infea_left[k] -= it->second;
						infea += fabs(infea_left[k]);
						
						msg_left_i2[k] -= delta_mu_alpha;
					}
					for (vector<pair<Int, Float>>::iterator it = act_k_index[i1+1].begin(); it != act_k_index[i1+1].end(); it++){
						Int k = it->first;
						Float delta_mu_alpha = admm_step_size*it->second;

						infea -= fabs(infea_right[k]);
						infea_right[k] -= it->second;
						infea += fabs(infea_right[k]);
						
						msg_right_i2[k] -= delta_mu_alpha;
					}
				}
			}
			
			admm_maintain_time += get_current_time();
			infea /= (2*M);
			
			Float nnz_alpha=0;
			for(Int i=0;i<N;i++){
				nnz_alpha += act_k_index[i].size();
			}
			nnz_alpha /= N;
			
			Float nnz_beta=0;
			for(Int i=0;i<M;i++){
				nnz_beta += act_kk_index[i].size();
			}
			nnz_beta /= M;
			
			cerr << "i=" << iter;
			cerr << ", infea=" << infea;
			cerr << ", nnz_a=" << nnz_alpha << ", nnz_b=" << nnz_beta ;
			cerr << ", uni_search=" << uni_search_time << ", uni_subSolve=" << uni_subSolve_time << ", uni_maintain=" << uni_maintain_time ;
			cerr << ", bi_search="  << bi_search_time   << ", bi_subSolve="  << bi_subSolve_time << ", bi_maintain="  << bi_maintain_time ;
			cerr << ", admm_maintain=" << admm_maintain_time ;
			cerr << ", area1=" << (double)submat_top/submat_bottom << ", area23=" << (double)line_top/line_bottom << ", area4=" << (double)mat_top/mat_bottom;
			cerr << ", dual_obj=" << dual_obj();
			if ((iter+1) % heldout_period == 0){
				overall_time += get_current_time();
				Int subsample = heldout_prob->data.size();
				cerr << ", p_obj=" << primal_obj(par, nSeq, subsample, model);
				overall_time -= get_current_time();
			}
			
			if ((iter+1) % write_model_period == 0){
				overall_time += get_current_time();
				Model* model = new Model(w, v, prob);
				char* name = new char[FNAME_LEN];
				sprintf(name, "%s.%d", modelFname, (iter+1));
				model->writeModel(name);
				overall_time -= get_current_time();
			}
			if ((iter+1) % heldout_period == 0 && heldout_prob != NULL){
				overall_time += get_current_time();
				Model* model = new Model(w, v, prob);
				Float heldout_test_acc = model->calcAcc_Viterbi(heldout_prob);
				cerr << ", heldout Acc=" <<  heldout_test_acc;
				overall_time -= get_current_time();
				if ( heldout_test_acc > max_heldout_test_acc){
					max_heldout_test_acc = heldout_test_acc;
					terminate_counting = 0;
				} else {
					cerr << " (" << (++terminate_counting) << "/" << (early_terminate) << ")";
					if (terminate_counting == early_terminate){
						//need to write final model
						if ((iter+1) % write_model_period != 0){
							overall_time += get_current_time();
							Model* model = new Model(w, v, prob);
							char* name = new char[FNAME_LEN];
							sprintf(name, "%s.%d", modelFname, (iter+1));
							model->writeModel(name);
							overall_time -= get_current_time();
						}
						cerr << endl;
						break;
					}
				}
			}
			cerr << ", overall time=" << (overall_time+get_current_time()) << endl;
			//if( p_inf < 1e-4 )
			//	break;
			
			//cerr << "i=" << iter << ", Acc=" << train_acc_Viterbi() << ", dual_obj=" << dual_obj() << endl;
		}

		delete[] uni_ind;
		delete[] bi_ind;
		delete[] alpha_new;
		delete[] beta_new;
		
		//search
		for (int i = 0; i < N; i++){
			act_k_index[i].clear();
		}
		for (int i = 0; i < M; i++){
			act_kk_index[i].clear();
		}
		delete[] act_k_index;
		delete[] act_kk_index;
		
		return new Model(w,v,prob);
	}

	private:

	void uni_update(Int i, Int n, Int t, Int il, Int ir, vector<pair<Int, Float>>& act_uni_index, Float* alpha_new, Float loss_per_node, Int iter){
		
		//data
		Seq* seq = data->at(n);
		Int yi = seq->labels[t];
		assert(yi == act_uni_index[0].first);
		SparseVec* xi = seq->features[t];
		//variable values
		Float Qii = Q_diag[i] + 2 * eta;
		//Float* alpha_i = alpha[i];
		
		Float* msg_to_left = zero_msg;
		Float* msg_to_right = zero_msg;
		if (ir != -1)
			msg_to_right = msg_left[ir];
		if (il != -1)
			msg_to_left = msg_right[il];
		Int size_grad_heap = act_uni_index.size();
		for(Int it = 0; it < size_grad_heap; it++){
			pair<Int, Float> p = act_uni_index[it];
			Int k = p.first;
			prod[it] = loss_per_node - eta*(msg_to_right[k]+msg_to_left[k]);
		}
		prod[0] -= loss_per_node;

		//compute gradient (bottleneck is here)
		for(SparseVec::iterator itt=xi->begin(); itt!=xi->end(); itt++){
			Int f_ind = itt->first;
			Float f_val = itt->second;
			for(Int it = 0; it < size_grad_heap; it++){
				Int k = act_uni_index[it].first;
				prod[it] += w[ f_ind ][k] * f_val;
			}
		}

		Int max_neg = -1;
		Float max_val_neg = -INFI;
		for(Int it = 1; it < size_grad_heap; it++){
			if (prod[it] > max_val_neg){
				max_neg = act_uni_index[it].first;
				max_val_neg = prod[it];
			}
		}
		Float obj = (prod[0] - max_val_neg) * C;
		if (max_neg == -1 || obj >= 0.0){
			//s_k = all zero vector
			for (vector<pair<Int, Float>>::iterator it_a = act_uni_index.begin(); it_a != act_uni_index.end(); it_a++){
				alpha_new[it_a->first] = 0.0;
			}
		} else {
			//s_k = all zero but yi and max_neg
			for (vector<pair<Int, Float>>::iterator it_a = act_uni_index.begin(); it_a != act_uni_index.end(); it_a++){
				alpha_new[it_a->first] = 0.0;
			}
			alpha_new[yi] += C;
			alpha_new[max_neg] -= C;
		}

		//compute gamma
		Float gamma = 0.0;
		Float up = 0.0, down = 0.0;
		Int ind = 0;
		for (vector<pair<Int, Float>>::iterator it_a = act_uni_index.begin(); it_a != act_uni_index.end(); it_a++){
			Int k = it_a->first;
			down += (Qii + 2*eta) * (alpha_new[k] - it_a->second) * (alpha_new[k] - it_a->second);
			/*for(SparseVec::iterator itt=xi->begin(); itt!=xi->end(); itt++){
				Int f_ind = itt->first;
				Float f_val = itt->second;
				up += (it_a->second-alpha_new[k]) * w[ f_ind ][k] * f_val;
			}
			if (k != yi)
				up += (it_a->second - alpha_new[k]) * loss_per_node;
			up -= eta * (msg_to_right[k] + msg_to_left[k]) * (it_a->second - alpha_new[k]);*/
			up -= prod[ind] * (alpha_new[k] - it_a->second);
			ind++;
		}
		if (fabs(down) > 1e-12)
			gamma = up / down;
		else
			gamma = 0.0;
		if (gamma < 0.0)
			gamma = 0.0;
		if (gamma > 1.0)
			gamma = 1.0;
		//gamma = 2.0 / (iter + 4.0);
	
		for (vector<pair<Int, Float>>::iterator it_a = act_uni_index.begin(); it_a != act_uni_index.end(); it_a++){
			Int k = it_a->first;
			alpha_new[k] = gamma * alpha_new[k] + (1-gamma)*it_a->second;
		}
		for (vector<pair<Int, Float>>::iterator it_a = act_uni_index.begin(); it_a != act_uni_index.end(); it_a++){
			Int k = it_a->first;
			//cout << prod[0] << " " << max_val_neg << " " << alpha_new[k] << endl;
			if (k == yi)
				assert(alpha_new[k] >= 0.0 && alpha_new[k] <= (Float)C);
			else
				assert(alpha_new[k] <= 0.0 && alpha_new[k] >= -(Float)C);
		}
		
	}	

	void uni_subSolve(Int i, Int n, Int t, Int il, Int ir, vector<pair<Int, Float>>& act_uni_index, Float* alpha_new, Float loss_per_node ){ //solve i-th unigram factor
		//memset(prod, 0.0, sizeof(Float)*K);
		//data
		Seq* seq = data->at(n);
		Int yi = seq->labels[t];
		assert(yi == act_uni_index[0].first);
		SparseVec* xi = seq->features[t];
		//variable values
		Float Qii = Q_diag[i] + 2 * eta;
		//Float* alpha_i = alpha[i];
		/*if( t!=0 )//not beginning
			msg_from_left = messages[2*bi_index(n,t-1)+F_RIGHT];
		if( t!=seq->T-1 ) //not end
			msg_from_right = messages[2*bi_index(n,t)+F_LEFT];
		*/
		
		Float* msg_to_left = zero_msg;
		Float* msg_to_right = zero_msg;
		if (ir != -1)
			msg_to_right = msg_left[ir];
		if (il != -1)
			msg_to_left = msg_right[il];
		Int size_grad_heap = act_uni_index.size();
		for(Int it = 0; it < size_grad_heap; it++){
			pair<Int, Float> p = act_uni_index[it];
			Int k = p.first;
			prod[it] = loss_per_node - Qii*p.second - eta*(msg_to_right[k]+msg_to_left[k]);
		}
		prod[0] -= loss_per_node;

		/*for(vector<pair<Int, Float>>::iterator it = act_uni_index.begin(); it != act_uni_index.end(); it++){
			Int k = it->first;
			//if( k!=yi )
			prod[k] = 1.0 - Qii*it->second;
			
			//else
			//	prod[k] = -Qii*alpha_i[k];
		}
		prod[yi] -= 1.0;
		//message=(E\beta-\alpha+\mu/\eta)
		if( t != 0 ){
			Float* msg_to_left = msg_right[il];
			for(vector<pair<Int, Float>>::iterator it = act_uni_index.begin(); it != act_uni_index.end(); it++){
				Int k = it->first;
				prod[k] -= eta*msg_to_left[k];
			}
		}
		if( t != seq->T-1 ){
			Float* msg_to_right = msg_left[ir];
			for(vector<pair<Int, Float>>::iterator it = act_uni_index.begin(); it != act_uni_index.end(); it++){
				Int k = it->first;
				prod[k] -= eta*msg_to_right[k];
			}
		}*/
		//compute gradient (bottleneck is here)
		for(SparseVec::iterator itt=xi->begin(); itt!=xi->end(); itt++){
			Int f_ind = itt->first;
			Float f_val = itt->second;
			for(Int it = 0; it < size_grad_heap; it++){
				Int k = act_uni_index[it].first;
				prod[it] += w[ f_ind ][k] * f_val;
			}
		}
		
		//compute gradients
		for(Int ind = 0; ind < size_grad_heap; ind++){
			Int k = act_uni_index[ind].first;
			//if( k != yi ){
			grad_heap[ind] = prod[ind];
			//}else{
			//	grad_heap[ind] = prod[ind] + Qii*C;	
			//}
		}
		grad_heap[0] += Qii*C;
		
		//compute b by traversing gradients in descending order
		Float b; Int r;
		for (r = 0; r < size_grad_heap; r++){
			for (Int itt = r+1; itt < size_grad_heap; itt++){
				if (grad_heap[itt] > grad_heap[r]){
					Float temp = grad_heap[itt];
					grad_heap[itt] = grad_heap[r];
					grad_heap[r] = temp;
				}
			}
			if (r == 0){
				b = grad_heap[0] - Qii*C;
			} else {
				if (b >= grad_heap[r]*r){
					break;
				}
				b+=grad_heap[r];
			}
		}
		b = b / r;
		
		//record alpha new values
		for(Int it = 0; it < act_uni_index.size(); it++){
			Int k = act_uni_index[it].first;
			alpha_new[k] = min( (Float)((k!=yi)?0.0:C), (b-prod[it])/Qii );
		}

	}

	void uni_search(Int i, Int n, Int t, Int il, Int ir, vector<pair<Int, Float>>& act_k_index){
		Int rand_interval = rand() % split_up_rate;
		Int range_l = interval_start[rand_interval], range_r = interval_end[rand_interval];

		Seq* seq = data->at(n);
		Int yi = seq->labels[t];
		memset(prod, 0.0, sizeof(Float)*K);
		SparseVec* xi = seq->features[t];
		
		Float* msg_to_left = zero_msg;
		Float* msg_to_right = zero_msg;
		if (ir != -1)
			msg_to_right = msg_left[ir];
		if (il != -1)
			msg_to_left = msg_right[il];
		for (vector<pair<Int, Float>>::iterator it = act_k_index.begin(); it != act_k_index.end(); it++){
			prod[it->first] = -INFI;
		}
		max_indices[0] = range_l;
		for (Int k = range_l; k < range_r; k++){
			prod[k] -= eta*(msg_to_right[k] + msg_to_left[k]);
		}

		//Float th = -1.0/seq->T;
		Float th = -1.0;
		for (SparseVec::iterator it = xi->begin(); it != xi->end(); it++){
			Float xij = it->second;
			Int j = it->first;
			Float* wj = w[j];
			for (int k = range_l; k < range_r; k++){
				prod[k] += wj[k] * xij;
			}
		}
		
		max_indices[0] = range_l;
		for (Int k = range_l; k < range_r; k++){
			if (prod[k] > prod[max_indices[0]]){
				max_indices[0] = k;
			}
		}

		if (prod[max_indices[0]] > th){
			act_k_index.push_back(make_pair(max_indices[0], 0.0));
		}
	}
	
	void bi_update(Int i, Int n, Int t, vector<pair<Int, Float>>& act_bi_index, Float* beta_new){
			
		//data
		Seq* seq = data->at(n);
		Int yi = seq->labels[t];
		Int yj = seq->labels[t+1];
		Int yi_yj = yi*K + yj;

		//variable values
		Float* msg_from_left = msg_left[i];
		Float* msg_from_right = msg_right[i];
		
		//compute gradient
		size_grad_heap = act_bi_index.size();
		Int ind = 0;
		Int max_neg = -1;
		Float max_val_neg = -INFI;
		Float obj = 0.0;
		for (vector<pair<Int, Float>>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = it->first;
			Int k1 = k1k2 / K;
			Int k2 = k1k2 % K;
			grad[ind] = v[k1][k2]+eta*(msg_from_left[k1]+msg_from_right[k2]);
			if (k1k2 != yi_yj && grad[ind] > max_val_neg){
				max_neg = k1k2;
				max_val_neg = grad[ind];
			} 
			if (k1k2 == yi_yj) {
				obj += grad[ind];
			}
			ind++;
		}

		if (max_neg == -1 || obj - max_val_neg >= 0.0){
			//all zero vector
			for (vector<pair<Int, Float>>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
				Int k1k2 = it->first;
				beta_new[k1k2] = 0.0;
			}
		} else {
			//all zero vector except yi_yj and max_neg
			for (vector<pair<Int, Float>>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
				Int k1k2 = it->first;
				beta_new[k1k2] = 0.0;
			}
			beta_new[yi_yj] = C;
			beta_new[max_neg] = -C;
		}

		Float* beta_suml = new Float[K];		
		Float* beta_sumr = new Float[K];		
		
		memset(beta_suml, 0.0, sizeof(Float)*K);
		memset(beta_sumr, 0.0, sizeof(Float)*K);
		Float gamma = 0.0, down = 0.0, up = 0.0;
		for (vector<pair<Int, Float>>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = it->first;
			Int k1 = k1k2 / K, k2 = k1k2 % K;
			Float delta_beta = beta_new[k1k2] - it->second;
			down -= eta * beta_suml[k1] * beta_suml[k1];
			down -= eta * beta_sumr[k2] * beta_sumr[k2];
			beta_suml[k1] += delta_beta;
			beta_sumr[k2] += delta_beta;
			down += eta * beta_suml[k1] * beta_suml[k1];
			down += eta * beta_sumr[k2] * beta_sumr[k2];
			up -= eta * (msg_from_left[k1] + msg_from_right[k2]) * delta_beta;
			
		}
		
		for (vector<pair<Int, Float>>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = it->first;
			Int k1 = k1k2 / K, k2 = k1k2 % K;
			Float delta_beta = beta_new[k1k2] - it->second;
			up -= v[k1][k2] * delta_beta;
			down += delta_beta * delta_beta;
		}
		if (fabs(down) > 1e-12)
			gamma = up/down;
		if (gamma > 1.0) 
			gamma = 1.0;
		if (gamma < 0.0) 
			gamma = 0.0;
		//record alpha new values
		ind = 0;
		for (vector<pair<Int, Float>>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = it->first;
			beta_new[k1k2] = beta_new[k1k2] * gamma + (1 - gamma) * it->second; 
			ind++;
		}
		delete beta_suml;
		delete beta_sumr;
	}
	
	void bi_subSolve(Int i, Int n, Int t, vector<pair<Int, Float>>& act_bi_index, Float* beta_new){
			
		//data
		Seq* seq = data->at(n);
		Int yi = seq->labels[t];
		Int yj = seq->labels[t+1];
		Int yi_yj = yi*K + yj;
		Int il = uni_index(n, t);
		Int ir = uni_index(n, t+1);

		//variable values
		//Float* beta_i = beta[i];
		//Float* msg_from_left = messages[2*bi_index(n,t)+F_LEFT];
		//Float* msg_from_right = messages[2*bi_index(n,t)+F_RIGHT];
		Float* msg_from_left = msg_left[i];
		Float* msg_from_right = msg_right[i];
		
		//compute gradient
		Float Qii = (1.0+eta*(act_k_index[il].size() + act_k_index[ir].size()) );
		
			
		/*for(vector<Int>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = *it;
			grad[k1k2] = -Qii*beta_i[k1k2];
		}
	
		for(vector<Int>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = *it;
			Int k2 = k1k2 % K;
			Int k1 = k1k2 / K;
			grad[k1k2] += v[k1][k2];
			//grad: message from left
			grad[k1k2] += eta*msg_from_left[k1];
			//grad: message from right
			grad[k1k2] += eta*msg_from_right[k2];
		}
		size_grad_heap = act_bi_index.size();
		for (Int itt = 0; itt < act_bi_index.size(); itt++){
			Int k1k2 = act_bi_index[itt];
			//compute Dk
			if( k1k2 != yi_yj ){
				//Dk[itt] = grad[k1k2];
				grad_heap[itt] = grad[k1k2];
				//siftUp(grad_heap, itt);
			}else{
				//Dk[itt] = grad[k1k2] + Qii*C;
				grad_heap[itt] = grad[k1k2] + Qii*C;
				//siftUp(grad_heap, itt);
			}
		}*/

		size_grad_heap = act_bi_index.size();
		Int ind = 0;
		for (vector<pair<Int, Float>>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = it->first;
			Int k1 = k1k2 / K;
			Int k2 = k1k2 % K;
			grad[ind] = v[k1][k2]+eta*(msg_from_left[k1]+msg_from_right[k2])-Qii*it->second;
			if (k1k2 == yi_yj)
				grad_heap[ind] = grad[ind] + Qii*C;
			else
				grad_heap[ind] = grad[ind];
			ind++;
		}

		Float b; 
		/*if (size_grad_heap > 50){
			for (Int itt = (size_grad_heap-1)/2; itt >= 0; itt--){
				siftDown(grad_heap, itt, size_grad_heap);
			}
			Int r = 1;
			b = grad_heap[0] - Qii*C;
			grad_heap[0] = grad_heap[--size_grad_heap];
			siftDown(grad_heap, 0, size_grad_heap);
			while (size_grad_heap > 0){
				Float next = grad_heap[0];
				if (b >= next*r)
					break;
				b+=next; r++;
				grad_heap[0] = grad_heap[--size_grad_heap];
				siftDown(grad_heap, 0, size_grad_heap);
			}
			b = b / r;
		} else {*/
			Int r;
			for (r = 0; r < size_grad_heap; r++){
				for (Int itt = r+1; itt < size_grad_heap; itt++){
					if (grad_heap[itt] > grad_heap[r]){
						Float temp = grad_heap[itt];
						grad_heap[itt] = grad_heap[r];
						grad_heap[r] = temp;
					}
				}
				if (r == 0){
					b = grad_heap[0] - Qii*C;
				} else {
					if (b >= grad_heap[r]*r)
						break;
					b+=grad_heap[r]; 
				}
			}
			b = b / r;
		//}
		
		//record alpha new values
		ind = 0;
		for (vector<pair<Int, Float>>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = it->first;
			beta_new[k1k2] = min( (Float)((k1k2!=yi_yj)?0.0:C), (b-grad[ind])/Qii );
			ind++;
		}
		/*for(vector<Int>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = *it;
			beta_new[k1k2] = min( (Float)((k1k2!=yi_yj)?0.0:C), (b-grad[k1k2])/Qii );
		}*/
		
	}
	
	void bi_search(Int i, Int n, Int t, vector<pair<Int, Float>>& act_bi_index){

		for (vector<pair<Int, Float>>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			inside[it->first] = true;
		}
		
		Int il = uni_index(n, t);
		Int ir = uni_index(n, t+1);
		Int max_k1k2 = -1;
		Float max_val = -1e300;
		
		//compute messages and then sort in decreasing order
		/*Float* msg_from_left = new Float[K];
		Float* msg_from_right = new Float[K];
		memset(msg_from_left, 0.0, sizeof(Float)*K);
		memset(msg_from_right, 0.0, sizeof(Float)*K);
		
		Float* beta_suml_i = beta_suml[i];
		Float* beta_sumr_i = beta_sumr[i];
		Float* alpha_il = alpha[il];
		Float* alpha_ir = alpha[ir];
		Float* mu_il = mu[i*2];
		Float* mu_ir = mu[i*2+1];*/

		//vector<Int> pos_left;	
		//vector<Int> pos_right;
		Float* msg_from_left  = msg_left[i];
		Float* msg_from_right = msg_right[i];
			
		pos_left.clear(); pos_right.clear();
		for (int k = 0; k < K; k++){
			//msg_from_left[k] = beta_suml_i[k] - alpha_il[k] + mu_il[k];
			//msg_from_right[k] = beta_sumr_i[k] - alpha_ir[k] + mu_ir[k];
			//assert(fabs(beta_suml_i[k] - alpha_il[k] + mu_il[k] - msg_from_left[k])<1e-3 );
			//assert(fabs(beta_sumr_i[k] - alpha_ir[k] + mu_ir[k] - msg_from_right[k])<1e-3 );
			if (msg_from_left[k] > 0.0)
				pos_left.push_back(k);
			if (msg_from_right[k] > 0.0)
				pos_right.push_back(k);
		}
		//sort(left_index, left_index+K, ScoreComp(msg_to_left));
		//sort(right_index, right_index+K, ScoreComp(msg_to_right));
		
		//check area 1: msg_to_left > 0 and msg_to_right > 0
			
		submat_bottom += 1;
		submat_top += pos_left.size()*pos_right.size();	
		for (vector<Int>::iterator it_l = pos_left.begin(); it_l != pos_left.end(); it_l++){
			int kl = *it_l;
			Float* v_kl = v[kl];
			Float msg_L_kl = msg_from_left[kl];
			for (vector<Int>::iterator it_r = pos_right.begin(); it_r != pos_right.end(); it_r++){
				int kr = *it_r;
				Float val = v_kl[kr] + eta*(msg_L_kl + msg_from_right[kr]);
				if (val > max_val && !inside[kl*K+kr]){
					max_val = val;
					max_k1k2 = kl*K+kr;
				}
			}
		}
			
		//check area 2: msg_to_left > 0 and msg_to_right = 0
		for (vector<Int>::iterator it_l = pos_left.begin(); it_l != pos_left.end(); it_l++){
			Int k1 = *it_l;
			search_line(row_heap[k1], msg_from_left[k1], msg_from_right, max_val, max_k1k2, row_heap_size[k1], inside, col_dir, eta);
		}
		
		//check area 3: msg_to_left = 0 and msg_to_right > 0
		for (vector<Int>::iterator it_r = pos_right.begin(); it_r != pos_right.end(); it_r++){
			Int k2 = *it_r;
			search_line(col_heap[k2], msg_from_right[k2], msg_from_left, max_val, max_k1k2, col_heap_size[k2], inside, row_dir, eta);
		}
		
		//check area 4: msg_to_left <= 0 and msg_to_right <= 0	
		search_matrix(v_heap, msg_from_left, msg_from_right, max_val, max_k1k2, v_heap_size, inside, K, eta);
		
		for (vector<pair<Int, Float>>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			inside[it->first] = false;
		}
		if (max_val > 0.0){
			//assert(fabs(real_max_val - max_val) < 1e-5);
			act_bi_index.push_back(make_pair(max_k1k2, 0.0));
		}
	}
	
	void bi_brute_force_search(Int i, Int n, Int t, vector<pair<Int, Float>>& act_bi_index){
		Int max_k1k2 = -1;
		Float max_val = -1e300;

		for (vector<pair<Int, Float>>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			inside[it->first] = true;
		}

		Float* msg_left_i = msg_left[i];
		Float* msg_right_i = msg_right[i];
		for(int k1 = 0; k1 < K; k1++){
			Float msg_left_i_k1 = msg_left_i[k1];
			Float* v_k1 = v[k1];
			for (int k2 = 0; k2 < K; k2++){
				if (inside[k1*K+k2]) continue;
				Float val = v_k1[k2]+eta*(msg_left_i_k1 + msg_right_i[k2]);
				if (val > max_val){
					max_k1k2 = k1*K+k2;
					max_val = val;
				}
			}
		}
		for (vector<pair<Int, Float>>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			inside[it->first] = false;
		}

		if (max_val > 0.0){
			act_bi_index.push_back(make_pair(max_k1k2, 0.0));
		}
	}

	//backup old vesion
	/*void bi_brute_force_search(Int i, Int n, Int t, vector<Int>& act_bi_index){
		Int il = uni_index(n, t);
		Int ir = uni_index(n, t+1);
		Int max_k1k2 = -1;
		Float max_val = -1e300;
		
		for(int k1 = 0; k1 < K; k1++){
			for (int k2 = 0; k2 < K; k2++){
				if (find(act_bi_index.begin(), act_bi_index.end(), k1*K+k2) != act_bi_index.end()){
					continue;
				}
				Float val = v[k1][k2];
				val += beta_suml[i][k1];
				val -= alpha[il][k1];
				
				val += beta_sumr[i][k2];
				val -= alpha[ir][k2];
				val += mu[i*2][k1];
				val += mu[i*2+1][k2];
				if (val > max_val){
					max_k1k2 = k1*K+k2;
					max_val = val;
				}
			}
		}

		if (max_val > 0.0){
			//real_max_val = max_val;
			act_bi_index.push_back(max_k1k2);
		}
	}*/
	void compute_offset(){
		
		uni_offset = new Int[nSeq];
		bi_offset = new Int[nSeq];
		uni_offset[0] = 0;
		bi_offset[0] = 0;
		for(Int i=1;i<nSeq;i++){
			uni_offset[i] = uni_offset[i-1] + data->at(i-1)->T;
			bi_offset[i] = bi_offset[i-1] + data->at(i-1)->T-1;
		}
	}

	inline Int uni_index(Int n, Int t){
		return uni_offset[n]+t;
	}
	inline Int bi_index(Int n, Int t){
		return bi_offset[n]+t;
	}
	inline void get_uni_rev_index(Int i, Int& n, Int& t){
		n=1;
		while( n<nSeq && i >= uni_offset[n] )n++;
		n -= 1;
		
		t = i-uni_offset[n];
	}
	inline void get_bi_rev_index(Int i, Int& n, Int& t){
		n=1;
		while( n<nSeq && i >= bi_offset[n] )n++;
		n -= 1;
		
		t = i - bi_offset[n];
	}
	
	Float train_acc_unigram(){
		
		Float* prod = new Float[K];
		Int hit=0;
		for(Int n=0;n<nSeq;n++){
			Seq* seq = data->at(n);
			for(Int t=0; t<seq->T; t++){
				
				SparseVec* xi = seq->features[t];
				Int yi = seq->labels[t];
				for(Int k=0;k<K;k++)
					prod[k] = 0.0;
				for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
					Float* wj = w[it->first];
					for(Int k=0;k<K;k++)
						prod[k] += wj[k]*it->second;
				}
				Float max_val = -1e300;
				Int argmax;
				for(Int k=0;k<K;k++)
					if( prod[k] > max_val ){
						max_val = prod[k];
						argmax = k;
					}
				
				if( argmax == yi )
					hit++;
			}
		}
		Float acc = (Float)hit/N;
		
		delete[] prod;
		return acc;
	}
	
	Float dual_obj(){
		
		Float uni_obj = 0.0;
		for(Int j=0;j<D;j++){
			for(Int k=0;k<K;k++){
				uni_obj += w[j][k] * w[j][k];
			}
		}
		uni_obj/=2.0;
		
		for(Int n=0;n<nSeq;n++){
			Seq* seq = data->at(n);
			for(Int t=0; t<seq->T; t++){
				Int i = uni_index(n,t);
				Int yi = seq->labels[t];
				Int len = seq->labels.size();
				for(vector<pair<Int, Float>>::iterator it = act_k_index[i].begin(); it != act_k_index[i].end(); it++){
					if (it->first != yi)
						uni_obj += it->second;
				}
				/*for(Int k=0;k<K;k++)
					if( k != yi ){
						uni_obj += alpha[i][k];
					}*/
			}
		}

		Float bi_obj = 0.0;
		for(Int j=0;j<K;j++){
			for(Int k=0;k<K;k++)
				bi_obj += v[j][k] * v[j][k];
		}
		bi_obj/=2.0;
			
		Float p_inf_ijk;
		Float p_inf = 0.0;
		Float* cache = new Float[K];
		Float* beta_suml = new Float[K];
		Float* beta_sumr = new Float[K];
		for(Int n=0;n<nSeq;n++){
			Seq* seq = data->at(n);
			for(Int t=0;t<seq->T-1;t++){
				Int i2 = bi_index(n,t);
				Int i1 = uni_index(n,t);
				memset(beta_suml, 0.0, sizeof(Float)*K);
				memset(beta_sumr, 0.0, sizeof(Float)*K);
				for (vector<pair<Int, Float>>::iterator it_b = act_kk_index[i2].begin(); it_b != act_kk_index[i2].end(); it_b++){
					Int k1k2 = it_b->first;
					Int k1 = k1k2 / K, k2 = k1k2 % K;
					beta_suml[k1] += it_b->second;
					beta_sumr[k2] += it_b->second;
				}
				for(Int j=0;j<NUM_DIRECT;j++){
					Float* mu_ij = mu[2*i2+j];
					memset(cache, 0.0, sizeof(Float)*K);
					for (vector<pair<Int, Float>>::iterator it = act_k_index[i1+j].begin(); it != act_k_index[i1+j].end(); it++){
						cache[it->first] = it->second;
					}
					for(Int k=0;k<K;k++){
						if (j == 0)
							p_inf_ijk = beta_suml[k] - cache[k];
						else
							p_inf_ijk = beta_sumr[k] - cache[k];
						p_inf += p_inf_ijk * p_inf_ijk;
					}
				}
			}
		}
		p_inf *= eta/2.0;
		delete[] cache;
		delete[] beta_suml;
		delete[] beta_sumr;
		return uni_obj + bi_obj + p_inf;
	}

	ChainProblem* prob;
	
	Param* par;
	Model* model;
	
	vector<Seq*>* data;
	Float C;
	Int nSeq;
	Int N; //number of unigram factors (#variables)
	Int M; //number of bigram factors
	Int D;
	Int K;
	Int* uni_offset;
	Int* bi_offset;
		
	Float* Q_diag;
	//active set for alpha
	vector<pair<Int, Float>>* act_k_index;
	//active set for beta
	vector<pair<Int, Float>>* act_kk_index;	
	//Float** alpha; //N*K dual variables for unigram factor
	//Float** beta; //M*K^2 dual variables for bigram factor
	Float** beta_suml;	
	Float** beta_sumr;	
	Float** w; //D*K primal variables for unigram factor
	Float** v; //K^2 primal variables for bigram factor


	Float** mu; // 2M*K Lagrangian Multipliers on consistency constraInts
	//Float** messages;// 2M*K message=(E*beta-alpha+\frac{1}{\eta}\mu)

	bool do_subSolve;	

	Float* h_left;
	Float* h_right;
	
	Int max_iter;
	Int write_model_period;
	char* modelFname;
	Float eta;
	Float admm_step_size;

	//heldout set
	ChainProblem* heldout_prob;
	//if heldout test accuracy doesn't increase in a few iterations, stop!
	Int early_terminate;
	Int heldout_period;

	//uni_search
	Int* max_indices;
	vector<Int>* w_nz_index;
	Int split_up_rate = 1;
	Int* interval_start;
	Int* interval_end;

	// oriented from beta nodes to its connected alpha nodes
	// msg_left[i2][k] = beta_suml[i2][k] - alpha[i2_left][k] + mu[i2+0][k]
	// msg_right[i2][k] = beta_sumr[i2][k] - alpha[i2_right][k] + mu[i2+1][k]
	Float** msg_left;
	Float** msg_right;	
	vector<Int> pos_left;
	vector<Int> pos_right;

	//global buffer
	Float* prod;
	Float* grad;
	Float* grad_heap;
	Int size_grad_heap;
	Float* zero_msg;

	//bi_search
	ArrayHeap* row_heap;
	ArrayHeap* col_heap;
	ArrayHeap v_heap;
	Int* row_heap_size;
	Int* col_heap_size;
	Int v_heap_size;
	Int* col_index;
	Int* row_index;
	Int* v_index;
	Int* col_dir;
	Int* row_dir;
	bool* inside;
	bool using_brute_force;
	double area1_time, area23_time, area4_time;
};
