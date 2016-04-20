#include "util.h"

class Heap{
	
	public:
	pair<Float, Int>* heap;
	Int* rev_index;
	Int size_heap;
	Heap(Int _size_heap){
		size_heap = _size_heap;
		heap = new pair<Float, Int>[size_heap];
		rev_index = new Int[size_heap];
	}

	Heap(Int _size_heap, Float* data){
		size_heap = _size_heap;
		heap = new pair<Float, Int>[size_heap];
		rev_index = new Int[size_heap];
		for (Int i = 0; i < size_heap; i++){
			heap[i] = make_pair(data[i], i);
			rev_index[i] = i;
		}
	}

	~Heap(){
		delete[] rev_index;
		delete[] heap;
	}
	
	//sift up, maintain reverse index
	inline void siftUp(Int index){
		pair<Float, Int> cur = heap[index];
		while (index > 0){
			Int parent = (index-1) >> 1;
			if (cur > heap[parent]){
				heap[index] = heap[parent];
				rev_index[heap[parent].second] = index;
				index = parent;
			} else {
				break;
			}
		}
		rev_index[cur.second] = index;
		heap[index] = cur;
	}

	//sift down, maintain reverse index
	inline void siftDown(Int index){
		pair<Float, Int> cur = heap[index];
		Int lchild = index * 2 +1;
		Int rchild = lchild+1;
		while (lchild < size_heap){
			Int next_index = index;
			if (heap[lchild] > heap[index]){
				next_index = lchild;
			}
			if (rchild < size_heap && heap[rchild] > heap[next_index]){
				next_index = rchild;
			}
			if (index == next_index) 
				break;
			heap[index] = heap[next_index];
			rev_index[heap[index].second] = index;
			heap[next_index] = cur;
			index = next_index;
			lchild = index * 2 +1; rchild = lchild+1;
		}
		rev_index[cur.second] = index;
	}
}
