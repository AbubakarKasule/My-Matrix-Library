#ifndef __MySparseMatrix_h__
#define __MySparseMatrix_h__

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>
#include "MyMatrix.h"

using namespace std;

class SparseMatrix
{
public:
	int m;						////number of rows
	int n;						////number of columns

	////HW3 Task 0: memory storage for a sparse matrix
	/*Your implementation starts*/
	vector<int> rows;
	vector<int> columns;
	vector<double> values;
	/*Your implementation ends*/

	////matrix constructor
	SparseMatrix(const int _m=1,const int _n=1)
	{
		Resize(_m,_n);	
	}

	void Resize(const int _m,const int _n)
	{
		m=_m;
		n=_n;	
	}

	////A=B
	void operator = (const SparseMatrix& B)
	{
		Resize(B.m,B.n);

		////HW3 Task 0: copy the data from another sparse matrix
		/*Your implementation starts*/
		rows = B.rows;
		columns = B.columns;
		values = B.values;
		/*Your implementation ends*/
	}

	////display matrix in terminal
	friend ostream & operator << (ostream &out,const SparseMatrix &mtx)
	{
		for(int i=0;i<mtx.m;i++){
			for(int j=0;j<mtx.n;j++){
				out<<mtx(i,j)<<", ";
			}
			out<<std::endl;
		}
		return out;
	}

	//////////////////////////////////////////////////////////////////////////
	////HW3 Task 1: initialize the three arrays using an array of tuples 
	void operator = (vector<tuple<int,int,double> >& elements)
	{
		//sort the elements by 1) row index i; 2) col index j
		sort(elements.begin(),elements.end(),
			[](const tuple<int,int,double>& a,const tuple<int,int,double>& b)->bool 
			{return get<0>(a)<get<0>(b)||(get<0>(a)==get<0>(b)&&get<1>(a)<=std::get<1>(b));});

		//Using the information in elements to initialize row, col, val
		//You may uncomment the following code to start or implement your own
		int r=-1;
		int valuesInPreviousRows = 0;
		for(int p=0;p<elements.size();p++){
			int i=get<0>(elements[p]);		////access row index i in tuple
			int j=get<1>(elements[p]);		////access j in tuple
			double v=get<2>(elements[p]);	////access value in tuple

		//	/*Your code starts*/
			values.push_back(v);
			columns.push_back(j);

			// Only add new element in rows if the item that we are on is on a new row than the previous item
			if (i > r){
				rows.push_back(valuesInPreviousRows);
				r = i;
			}
			valuesInPreviousRows++;
		//	/*Your code ends*/
		}
		///*Your code starts: for the last element!*/
		rows.push_back(valuesInPreviousRows);
		///*Your code ends*/
		
	}
	
	////HW3, Task 2: random access of a matrix element
	////notice that we are not using a reference in this case
	double operator() (int i,int j) const
	{
		assert(i>=0&&i<m&&j>=0&&j<n);
		////HW3: your implementation, random access element (i,j) by using row,col,val
		/*Your code starts*/
		for (int _i = rows[i]; _i < rows[i + 1]; _i++){
			// Find the corresponding index in columns vector
			if(columns[_i]==j){
				return values[_i];
			}
			else if (columns[_i] > j)    // We have gone past the possible location of a non-zero value
			{
				return 0;                // element must be a zero
			}
			

		}
		/*Your code ends*/
		return 0;
	}

	////HW3, Task 3: sparse matrix-vector multiplication
	////implement sparse prod=Ax, assuming x is a vector (with n==1)
	Matrix operator * (const Matrix& x)
	{
		assert(x.n==1 && x.m == n);
		Matrix prod(x.m,1);


		/*Your code starts*/
		for (int i = 0; i < m; i++)
		{
			for (int _i = rows[i]; _i < rows[i + 1]; _i++)
			{
				prod(i, 0) += (double) values[_i] * x(columns[_i], 0);
			}
			
		}
		/*Your code ends*/

		return prod;
	}
};

#endif