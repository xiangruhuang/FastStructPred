#include "util.h"
#include "chain.h"

class BDMMsolve{
	
	public:
	enum Direction {F_LEFT=0, F_RIGHT=1, NUM_DIRECT};
	
	BDMMsolve(Param* param){
		
		//Parse info from ChainProblem
		prob = param->prob;
		heldout_prob = param->heldout_prob;
		data = &(prob->data);
		nSeq = data->size();
		D = prob->D;
		K = prob->K;
		
		C = param->C;
		eta = param->eta;
		max_iter = param->max_iter;
		admm_step_size = param->admm_step_size;
		early_terminate = param->early_terminate;
		if (early_terminate <= 0)
			early_terminate = 3;	
	
		//Compute unigram and bigram offset[i] = \sum_{j=1}^{i-1} T_j
		compute_offset();
		N = uni_offset[nSeq-1] + data->at(nSeq-1)->T; //#unigram factor
		M = bi_offset[nSeq-1] + data->at(nSeq-1)->T - 1; //#bigram factor
		//allocate dual variables
		alpha = new Float*[N];
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
			for(Int k2=0;k2<K;k2++)
				v[k][k2] = 0.0;
		}
		
		//allocating Lagrangian Multipliers for consistency constraInts
		mu = new Float*[2*M]; //2 because of bigram
		//messages = new Float*[2*M];
		for(Int i=0;i<2*M;i++){
			mu[i] = new Float[K];
			//messages[i] = new Float[K];
			for(Int k=0;k<K;k++)
				mu[i][k] = 0.0;
			//for(Int k=0;k<K;k++)
			//	messages[i][k] = 0.0;
		}
		
		//pre-allocate some algorithmic constants
		Q_diag = new Float[N];
		for(Int n=0;n<nSeq;n++){
			Seq* seq = data->at(n);
			for(Int t=0;t<seq->T;t++){
				Int i = uni_index( n, t );
				Q_diag[i] = eta;
				for(SparseVec::iterator it=seq->features[t]->begin(); it!=seq->features[t]->end(); it++){
					Q_diag[i] += it->second * it->second;
				}
			}
		}
	}
	
	~BDMMsolve(){
		
		delete[] uni_offset;
		delete[] bi_offset;
		//dual variables
		for(Int i=0;i<N;i++)
			delete[] alpha[i];
		delete[] alpha;
		for(Int i=0;i<M;i++)
			delete[] beta[i];
		delete[] beta;
		//primal variables
		for(Int j=0;j<D;j++)
			delete[] w[j];
		delete[] w;
		for(Int k=0;k<K;k++)
			delete[] v[k];
		delete[] v;
		//Lagrangian Multiplier for Consistency constarInts
		for(Int i=0;i<2*M;i++){
			delete[] mu[i];
			//delete[] messages[i];
		}
		delete[] mu;
		//delete[] messages;
		
		//some constants
		delete Q_diag;
	}
	
	Model* solve(){
		
		Int* uni_ind = new Int[N];
		for(Int i=0;i<N;i++)
			uni_ind[i] = i;
		Int* bi_ind = new Int[M];
		for(Int i=0;i<M;i++)
			bi_ind[i] = i;
		
		//BDMM main loop
		Float* alpha_new = new Float[K];
		Float* beta_new = new Float[K*K];
		Float* marg_ij = new Float[K];
		Float p_inf;
		Float max_heldout_test_acc = 0.0;
		Int terminate_counting = 0;
		for(Int iter=0;iter<max_iter;iter++){
			
			random_shuffle(uni_ind, uni_ind+N);
			//update unigram dual variables
			for(Int r=0;r<N;r++){

				Int i = uni_ind[r];
				Int n, t;
				get_uni_rev_index(i, n, t);
				Seq* seq = data->at(n);
				
				//subproblem solving
				Float loss_per_node = 1.0/seq->T;
				uni_subSolve(i, n, t, alpha_new, loss_per_node);
				//maIntain relationship between w and alpha
				Float* alpha_i = alpha[i];
				SparseVec* xi = seq->features[t];
				for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
					Int j = it->first;
					Float fval = it->second;
					Float* wj = w[j];
					for(Int k=0;k<K;k++){
						wj[k] += fval * (alpha_new[k]-alpha_i[k]);
					}
				}
				//maintain messages(alpha) = (E*beta-alpha+\frac{1}{eta}mu)
				/*if( t != 0 ){
					Float* msg_to_left = messages[2*bi_index(n,t-1)+F_RIGHT];
					for(Int k=0;k<K;k++)
						msg_to_left[k] -= alpha_new[k] - alpha_i[k];
				}
				if( t != seq->T-1 ){
					Float* msg_to_right = messages[2*bi_index(n,t)+F_LEFT];
					for(Int k=0;k<K;k++)
						msg_to_right[k] -= alpha_new[k] + alpha_i[k];
				}
				*/

				//update alpha
				for(Int k=0;k<K;k++){
					alpha_i[k] = alpha_new[k];
				}

			}
			
			//update bigram dual variables	
			random_shuffle(bi_ind, bi_ind+M);
			for(Int r=0;r<M;r++){
				Int i = bi_ind[r];
				Int n, t;
				get_bi_rev_index(i, n, t);
				//subproblem solving
				bi_subSolve(i, n, t, beta_new);

				//maIntain relationship between v and beta
				Float* beta_i = beta[i];
				for(Int k=0;k<K;k++){
					Int Kk = K*k;
					for(Int k2=0;k2<K;k2++)
						v[k][k2] += beta_new[Kk+k2] - beta_i[Kk+k2];
				}
				//maintain messages(beta) = (E*beta-alpha+\frac{1}{eta}mu)
				/*Float* msg_to_left = messages[2*bi_index(n,t)+F_LEFT];
				Float* msg_to_right = messages[2*bi_index(n,t)+F_RIGHT];
				for(Int k=0; k<K; k++){
					Int Kk = K*k;
					for(Int k2=0; k2<K; k2++){
						msg_to_left[k] += beta_new[Kk+k2] - beta_i[Kk+k2];
						msg_to_right[k2] += beta_new[Kk+k2] - beta_i[Kk+k2];
					}
				}
				*/

				//update beta
				for(Int k=0;k<K;k++){
					Int Kk = K*k;
					for(Int k2=0;k2<K;k2++)
						beta_i[Kk+k2] = beta_new[Kk+k2];
				}
			}
			
			
			//ADMM update (enforcing consistency)
			Float mu_ijk;
			Float p_inf_ijk;
			p_inf = 0.0;
			for(Int n=0;n<nSeq;n++){
				Seq* seq = data->at(n);
				for(Int t=0;t<seq->T-1;t++){
					Int i2 = bi_index(n,t);
					Int i1 = uni_index(n,t);
					for(Int j=0;j<NUM_DIRECT;j++){
						Float* mu_ij = mu[2*i2+j];
						//Float* msg_ij = messages[2*i2+j];
						marginalize(beta[i2], (Direction)j, marg_ij);
						for(Int k=0;k<K;k++){
							//mu_ijk = mu_ij[k];
							p_inf_ijk = marg_ij[k] - alpha[i1+j][k];
							//p_inf_ijk = msg_ij[k] - mu_ijk;
							//update
							mu_ij[k] += admm_step_size*(p_inf_ijk);
							//maintain messages(mu) = (E*beta-alpha+\frac{1}{eta}mu)
							//msg_ij[k] += mu_ij[k] - mu_ijk;
							//compute infeasibility of consistency constraInt
							p_inf += fabs(p_inf_ijk);
						}
					}
				}
			}
			p_inf /= (2*M);
			
			double beta_nnz=0.0;
			for(Int i=0;i<M;i++){
				for(Int k=0;k<K;k++){
					for(Int k2=0;k2<K;k2++){
						if( fabs(beta[i][k*K+k2]) > 1e-6 )
							beta_nnz+=1.0;
					}
				}
			}
			beta_nnz/=M;
			
			double alpha_nnz=0.0;
			for(Int i=0;i<N;i++){
				for(Int k=0;k<K;k++){
					if( fabs(alpha[i][k]) > 0.0 )
						alpha_nnz += 1.0;
				}
			}
			alpha_nnz/=N;
			
			cerr << "i=" << iter << ", infea=" << p_inf << ", nnz_a=" << alpha_nnz << ", nnz_b=" << beta_nnz;
			cerr << ", dual_obj=" << dual_obj();
			if ((iter+1) % 1 == 0 && heldout_prob != NULL){	
				Model* model = new Model(w, v, prob);
				Float heldout_test_acc = model->calcAcc_Viterbi(heldout_prob);
				cerr << ", heldout Acc=" <<  heldout_test_acc;
				if ( heldout_test_acc > max_heldout_test_acc){
					max_heldout_test_acc = heldout_test_acc;
					terminate_counting = 0;
				} else {
					cerr << " (" << (++terminate_counting) << "/" << (early_terminate) << ")";
					if (terminate_counting == early_terminate){
						cerr << endl;
						break;	
					}
				}
			} else {
				cerr << ", train acc=" << train_acc_Viterbi();
			}
			cerr <<  endl;
			//if( p_inf < 1e-4 )
			//	break;
			
			//cerr << "i=" << iter << ", Acc=" << train_acc_Viterbi() << ", dual_obj=" << dual_obj() << endl;
		}
		
		delete[] marg_ij;
		delete[] uni_ind;
		delete[] bi_ind;
		delete[] alpha_new;
		delete[] beta_new;
		return new Model(w,v,prob);
	}

	private:
	
	void uni_subSolve(Int i, Int n, Int t, Float* alpha_new, Float loss_on_node){ //solve i-th unigram factor
		
		Float* grad = new Float[K];
		Float* Dk = new Float[K];
		
		//data
		Seq* seq = data->at(n);
		Int yi = seq->labels[t];
		SparseVec* xi = seq->features[t];
		//variable values
		Float Qii = Q_diag[i];
		Float* alpha_i = alpha[i];
		Float* msg_from_left=NULL;
		Float* msg_from_right=NULL;
		/*if( t!=0 )//not beginning
			msg_from_left = messages[2*bi_index(n,t-1)+F_RIGHT];
		if( t!=seq->T-1 ) //not end
			msg_from_right = messages[2*bi_index(n,t)+F_LEFT];
		*/
		if( t != 0 ){
			Int i2 = bi_index(n,t-1);
			msg_from_left = new Float[K];
			marginalize( beta[i2], F_RIGHT, msg_from_left);
			for(Int k=0;k<K;k++){
				msg_from_left[k] += -alpha[i][k] + mu[2*i2+F_RIGHT][k];
			}
		}
		if( t!=seq->T-1 ){
			Int i2 = bi_index(n,t);
			msg_from_right = new Float[K];
			marginalize( beta[i2], F_LEFT, msg_from_right );
			for(Int k=0;k<K;k++){
				msg_from_right[k] += -alpha[i][k] + mu[2*i2+F_LEFT][k];
			}
		}

		for(Int k=0;k<K;k++){
			if( k!=yi )
				grad[k] = loss_on_node - Qii*alpha_i[k];
			else
				grad[k] = -Qii*alpha_i[k];
		}
		//compute gradient (bottleneck is here)
		for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
			Int f_ind = it->first;
			Float f_val = it->second;
			for(Int k=0;k<K;k++)
				grad[k] += w[ f_ind ][k] * f_val ;
		}
		
		//message=(E\beta-\alpha+\mu/\eta)
		if( msg_from_left != NULL)
			for(Int k=0;k<K;k++)
				grad[k] -= eta*msg_from_left[k];
		
		if( msg_from_right != NULL )
			for(Int k=0;k<K;k++)
				grad[k] -= eta*msg_from_right[k];
		
		//compute Dk
		for(Int k=0;k<K;k++){
			if( k != yi )
				Dk[k] = grad[k];
			else
				Dk[k] = grad[k] + Qii*C;
		}

		//sort according to D_k
		sort( Dk, Dk+K, greater<Float>() );
		
		//compute b by traversing D_k in descending order
		Float bb = Dk[0] - Qii*C;
		Int r;
		for( r=1; r<K && bb<r*Dk[r]; r++)
			bb += Dk[r];
		bb = bb / r;
		
		//record alpha new values
		for(Int k=0;k<K;k++)
			alpha_new[k] = min( (Float)((k!=yi)?0.0:C), (bb-grad[k])/Qii );
		
		/*if( msg_from_left != NULL )
			delete[] msg_from_left;
		if( msg_from_right != NULL )
			delete[] msg_from_right;
		*/

		delete[] grad;
		delete[] Dk;
	}
	

	void bi_subSolve(Int i, Int n, Int t, Float* beta_new){
		
		Int Ksq = K*K;
		Float* grad = new Float[Ksq];
		Float* Dk = new Float[Ksq];
		
		//data
		Seq* seq = data->at(n);
		Int yi = seq->labels[t];
		Int yj = seq->labels[t+1];
		Int yi_yj = yi*K + yj;

		//variable values
		Float* beta_i = beta[i];
		//Float* msg_from_left = messages[2*bi_index(n,t)+F_LEFT];
		//Float* msg_from_right = messages[2*bi_index(n,t)+F_RIGHT];
		Float* msg_from_left = new Float[K];
		Float* msg_from_right = new Float[K];
		marginalize( beta_i, F_LEFT, msg_from_left );
		marginalize( beta_i, F_RIGHT, msg_from_right );
		Int i1 = uni_index(n,t);
		for(Int k=0;k<K;k++){
			msg_from_left[k] += -alpha[ i1 ][k] + mu[2*i+F_LEFT][k];
			msg_from_right[k] += -alpha[ i1+1 ][k] + mu[2*i+F_RIGHT][k];
		}
		
		//compute gradient
		Float Qii = (1.0+eta*K);
		for(Int k1k2=0; k1k2<Ksq; k1k2++)
			//if( k1k2 != yi_yj )
			//	grad[k1k2] = 1.0 - Qii*beta_i[k1k2];
			//else
				grad[k1k2] = -Qii*beta_i[k1k2];
		
		for(Int k1=0;k1<K;k1++){
			Int Kk1 = K*k1;
			for(Int k2=0;k2<K;k2++)
				grad[Kk1+k2] += v[k1][k2];
		}
		//grad: message from left
		for(Int k1=0;k1<K;k1++){
			Int Kk1 = k1*K;
			Float tmp = eta*msg_from_left[k1];
			for(Int k2=0;k2<K;k2++){
				grad[Kk1+k2] += tmp;
			}
		}
		//grad: message from right
		for(Int k1=0;k1<K;k1++){
			Int Kk1 = k1*K;
			for(Int k2=0;k2<K;k2++){
				grad[Kk1+k2] += eta*msg_from_right[k2];
			}
		}
		
		//compute Dk
		for(Int k1k2=0;k1k2<Ksq;k1k2++){
			if( k1k2 != yi_yj )
				Dk[k1k2] = grad[k1k2];
			else
				Dk[k1k2] = grad[k1k2] + Qii*C;
		}
		
		//sort according to D_k
		sort( Dk, Dk+Ksq, greater<Float>() );
		
		//compute b by traversing D_k in descending order
		Float b = Dk[0] - Qii*C;
		Int r;
		for( r=1; r<Ksq && b<r*Dk[r]; r++)
			b += Dk[r];
		b = b / r;
		
		//record alpha new values
		for(Int k1k2=0;k1k2<Ksq;k1k2++)
			beta_new[k1k2] = min( (Float)((k1k2!=yi_yj)?0.0:C), (b-grad[k1k2])/Qii );
		
		//// compute statistics
		double grad_max = -1e300;
		for(Int kk=0;kk<K*K;kk++){
			if( beta_i[kk] == 0.0 ){
				if( grad[kk] > grad_max )
					grad_max = grad[kk];
			}
		}
		double v_max = -1e300;
		Int argmax_k1;
		Int argmax_k2;
		for(Int k=0;k<K;k++){
			for(Int k2=0;k2<K;k2++){	
				if( beta_i[k*K+k2] == 0.0 && v[k][k2] > v_max ){
					v_max = v[k][k2];
					argmax_k1 = k;
					argmax_k2 = k2;
				}
			}
		}
		
		

		delete[] msg_from_left;
		delete[] msg_from_right;
		delete[] grad;
		delete[] Dk;
	}
	
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
	
	Float train_acc_Viterbi(){
		
		Int hit=0;
		for(Int n=0;n<nSeq;n++){
			
			Seq* seq = data->at(n);
			//compute prediction
			Int* pred = new Int[seq->T];
			Float** max_sum = new Float*[seq->T];
			Int** argmax_sum = new Int*[seq->T];
			for(Int t=0; t<seq->T; t++){
				max_sum[t] = new Float[K];
				argmax_sum[t] = new Int[K];
				for(Int k=0;k<K;k++)
					max_sum[t][k] = -1e300;
			}
			////Viterbi t=0
			SparseVec* xi = seq->features[0];
			for(Int k=0;k<K;k++)
				max_sum[0][k] = 0.0;
			for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
				Float* wj = w[it->first];
				for(Int k=0;k<K;k++)
					max_sum[0][k] += wj[k]*it->second;
			}
			////Viterbi t=1...T-1
			for(Int t=1; t<seq->T; t++){
				//passing message from t-1 to t
				for(Int k1=0;k1<K;k1++){
					Float tmp = max_sum[t-1][k1];
					Float cand_val;
					for(Int k2=0;k2<K;k2++){
						 cand_val = tmp + v[k1][k2];
						 if( cand_val > max_sum[t][k2] ){
							max_sum[t][k2] = cand_val;
							argmax_sum[t][k2] = k1;
						 }
					}
				}
				//adding unigram factor
				SparseVec* xi = seq->features[t];
				for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++)
					for(Int k2=0;k2<K;k2++)
						max_sum[t][k2] += w[it->first][k2] * it->second;
			}
			////Viterbi traceback
			pred[seq->T-1] = argmax( max_sum[seq->T-1], K );
			for(Int t=seq->T-1; t>=1; t--){
				pred[t-1] = argmax_sum[t][ pred[t] ];
			}
			
			//compute accuracy
			for(Int t=0;t<seq->T;t++){
				if( pred[t] == seq->labels[t] )
					hit++;
			}
			
			for(Int t=0; t<seq->T; t++){
				delete[] max_sum[t];
				delete[] argmax_sum[t];
			}
			delete[] max_sum;
			delete[] argmax_sum;
			delete[] pred;
		}
		Float acc = (Float)hit/N;
		
		return acc;
	}
	
	
	void marginalize( Float* table, Direction j, Float* marg ){
		
		for(Int k=0;k<K;k++)
			marg[k] = 0.0;
		
		if( j == F_LEFT ){
			for(Int k1=0;k1<K;k1++){
				Int Kk1 = K*k1;
				for(Int k2=0;k2<K;k2++)
					marg[k1] += table[Kk1+k2];
			}
		}else if( j == F_RIGHT ){
			for(Int k1=0;k1<K;k1++){
				Int Kk1 = K*k1;
				for(Int k2=0;k2<K;k2++)
					marg[k2] += table[Kk1+k2];
			}
		}else{
			cerr << "unknown direction: " << j << endl;
			exit(0);
		}
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
				for(Int k=0;k<K;k++)
					if( k != yi ){
						uni_obj += alpha[i][k]/seq->T;
					}
			}
		}
		
		Float bi_obj = 0.0;
		for(Int j=0;j<K;j++){
			for(Int k=0;k<K;k++)
				bi_obj += v[j][k] * v[j][k];
		}
		bi_obj/=2.0;
			
		Float p_inf_ijk;
		Float* marg_ij = new Float[K];
		Float p_inf = 0.0;
		for(Int n=0;n<nSeq;n++){
			Seq* seq = data->at(n);
			for(Int t=0;t<seq->T-1;t++){
				Int i2 = bi_index(n,t);
				Int i1 = uni_index(n,t);
				for(Int j=0;j<NUM_DIRECT;j++){
					//Float* msg_ij = messages[2*i2+j];
					//Float* mu_ij = mu[2*i2+j];
					marginalize(beta[i2], (Direction)j, marg_ij);
					for(Int k=0;k<K;k++){
						p_inf_ijk = marg_ij[k] - alpha[i1+j][k];
						//p_inf_ijk = msg_ij[k] - mu_ij[k];
						p_inf += p_inf_ijk * p_inf_ijk;
					}
				}
			}
		}
		p_inf *= eta/2.0;
		delete[] marg_ij;

		cerr << ", uni_obj=" << uni_obj << ", bi_obj=" << bi_obj << ", p_inf=" << p_inf;
	
		return uni_obj + bi_obj + p_inf;
	}


	ChainProblem* prob;
	
	//for heldout option
	ChainProblem* heldout_prob;
	Int early_terminate;
	
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
	Float** alpha; //N*K dual variables for unigram factor
	Float** beta; //M*K^2 dual variables for bigram factor
	
	Float** w; //D*K primal variables for unigram factor
	Float** v; //K^2 primal variables for bigram factor
	
	Float** mu; // 2M*K Lagrangian Multipliers on consistency constraInts
	//Float** messages;// 2M*K message=(E*beta-alpha+\frac{1}{\eta}\mu)
	
	Int max_iter;
	Float eta;
	Float admm_step_size;
	
};
