//////////////////////////////////////////////////////////////////////////
////Dartmouth CS70.01 sample code
////http://www.dartmouth.edu/~boolzhu/cosc70.01.html
////Linear algebra and vector math library
//////////////////////////////////////////////////////////////////////////

#ifndef __MyMatrix_h__
#define __MyMatrix_h__
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath> 

using namespace std;

class Matrix
{
public:
    int m;                        ////number of rows
    int n;                        ////number of columns
    vector<double> data;        ////element values, we use double for the data type

    ////matrix constructor
    Matrix(const int _m = 1, const int _n = 1)
    {
        Resize(_m, _n);
    }

    void Resize(const int _m, const int _n)
    {
        m = _m;
        n = _n;
        data.resize(m * n);
        for (int i = 0; i < m * n; i++) {
            data[i] = 0.;
        }
    }

    ////A=B
    void operator = (const Matrix& B)
    {
        Resize(B.m, B.n);

        for (int i = 0; i < (int)data.size(); i++) {
            data[i] = B.data[i];
        }
    }

    ////A={1.,4.,2.,...}, assigning a std::vector to A. A should initialized beforehand
    void operator = (const vector<double>& input_data)
    {
        assert(input_data.size() <= data.size());

        for (int i = 0; i < (int)input_data.size(); i++) {
            data[i] = input_data[i];
        }
    }

    ////return whether A==B
    bool operator == (const Matrix& B)
    {
        assert(m == B.m && n == B.n);
        for (int i = 0; i < (int)data.size(); i++)
        {
            if (abs(data[i] - B.data[i]) > 1e-6) return false;
        }
        return true;
    }

    ////return -A
    Matrix operator - ()
    {
        Matrix C(m, n);
        for (int i = 0; i < (int)data.size(); i++) {
            C.data[i] = -data[i];
        }
        return C;
    }

    ////random access of a matrix element
    double& operator() (int i, int j)
    {
        assert(i >= 0 && i < m && j >= 0 && j < n);
        return data[i * n + j];
    }

    const double& operator() (int i, int j) const
    {
        assert(i >= 0 && i < m && j >= 0 && j < n);
        return data[i * n + j];
    }

    ////display matrix in terminal
    friend ostream& operator << (ostream& out, const Matrix& mtx)
    {
        for (int i = 0; i < mtx.m; i++) {
            for (int j = 0; j < mtx.n; j++) {
                out << mtx(i, j) << ", ";
            }
            out << std::endl;
        }
        return out;
    }

    //////////////////////////////////////////////////////////////////////////
    ////overloaded operators

    ////matrix-matrix additions
    ////return C = A + B
    ////Notice: I use A to refer to the object itself in all my comments,
    ////if you want to self-access in the C++ code, you should use (*this),
    ////e.g., return (*this); means returning the object itself
    ////the comment A+=B; means (*this)+=B; in the code

    Matrix operator + (const Matrix& B)
    {
        assert(m == B.m && n == B.n);

        Matrix C(m, n);
        for (int i = 0; i < (int)data.size(); i++) {
            C.data[i] = data[i] + B.data[i];
        }
        return C;
    }

    ////A+=B
    void operator += (const Matrix& B)
    {
        assert(m == B.m && n == B.n);

        for (int i = 0; i < (int)data.size(); i++) {
            data[i] += B.data[i];
        }
    }

    //////////////////////////////////////////////////////////////////////////
    ////Your implementation starts

    ////Task 1: Mimic the "+" and "+=" operators,
    ////implement four new operators: "-", "-=", matrix-scalar multiplications "*" and "*="

    ////return A-B
    /*Your function implementation*/
    Matrix operator - (const Matrix& B)
    {
        assert(m == B.m && n == B.n);

        Matrix C(m, n);
        for (int i = 0; i < (int)C.data.size(); i++) {
            C.data[i] = data[i] - B.data[i];
        }
        return C;
    }

    ////A=A-B
    /*Your function implementation*/
    void operator -= (const Matrix& B)
    {
        assert(m == B.m && n == B.n);

        for (int i = 0; i < (int)data.size(); i++) {
            data[i] -= B.data[i];
        }
    }

    ////return A*s, with s as a scalar
    /*Your function implementation*/
    Matrix operator * (const double& s)
    {
        Matrix C(m, n);
        for (int i = 0; i < (int)data.size(); i++) {
            C.data[i] = data[i] * s;
        }
        return C;
    }

    ////A=A*s, with s as a scalar
    /*Your function implementation*/
    void operator *= (const double& s)
    {
        for (int i = 0; i < (int)data.size(); i++) {
            data[i] = data[i] * s;
        }
    }

    ////Task 2: matrix-matrix multiplication
    ////Hints: there are four steps:
    ////1, check compatibility by an assert;
    ////2, allocate a matrix C with proper size;
    ////3, calculate each element in C by a (left)row-(right)column multiplication
    //// when accessing an element (i,j) in the object itself, use (*this)(i,j)
    ////4, return c
    /*Your function implementation*/
    Matrix operator * (const Matrix& B)
    {
        assert(n == B.m);

        Matrix C(m, B.n);

        // loop through rows
        for (int column = 0; column < B.n; column++) {

            // loop through columns
            for (int row = 0; row < m; row++) {

                // initialize data to 0
                C.data[(row * B.n) + (column)] = 0;
                // get dot product for specified row and column
                for (int _n = 0; _n < n; _n++) {

                    C.data[(row * B.n) + (column)] += (data[_n + (row * n)] * B.data[(_n * B.n) + (column)]);
                }
            }
        }

        return C;
    }

    ////Task 3: identity, transpose(), block

    ////return an identity matrix
    /*Your function implementation*/
    Matrix Identity()
    {
        assert(m == n);
        Matrix C(m, n);

        for (int i = 0; i < n; i++) {
            C.data[(i * (n + 1))] = 1;
        }
        return C;
    }

    ////return A^T
    /*Your function implementation*/
    // I created the function item at this point to quickly access items in my Matrix
    Matrix Transpose()
    {
        Matrix C(n, m);

        for (int _n = 0; _n < n; _n++) {
            for (int _m = 0; _m < m; _m++) {
                C(_n, _m) = (*this)(_m, _n);
            }
        }
        return C;
    }

    ////return a submatrix block A_ijab,
    ////with i,j as the starting element and a,b as the block size
    /*Your function implementation*/
    Matrix Block(const int i, const int j, const int a, const int b)
    {
        Matrix C(a, b);
        
        for (int r = i; r < (i + a); r++) {
            for (int c = j; c < (j + b); c++) {
                C.data[item((r - i), (c - j))] = data[item(r, c)];
            }
        }
        return C;
    }

    ////Task 4: implement a function or a set of functions that were not specified in class
    int item(int i, int j) {
        assert(((i <= m) && (j <= n)) && ((i >= 0) && (j >= 0)));
        int locationIndex;

        locationIndex = ((n * i) + j);

        return locationIndex;
    }

    // Row Swap helper function
    Matrix Swap(Matrix A, int row_1, int row_2)
    {
        Matrix temp = A;
        for (int i = 0; i < A.n; i++){
            temp.data[A.item(row_2, i)] = A.data[A.item(row_1, i)];
            temp.data[A.item(row_1, i)] = A.data[A.item(row_2, i)];
        }
        return temp;
    }

    //////////////////////////////////////////////////////////////////////////
    ////Assignment 2: Solve Linear Equations using Gaussian Elimination
    ////Your implementation starts
    ////return the solution of linear equations Ax=b
    Matrix Solve(const Matrix& A, const Matrix& b, Matrix& solution)
    {
        //Compatibility Checks
		assert(A.n == b.m && b.n == 1);
		
		//initialize temporary variables
		// Matrix solution;
		Matrix U;
        Matrix bTemp;
		U = A;
        bTemp = b;
		double temp;
        int maxPivot;
        int maxPivotRow;
        double beta;
		////Forward Elimination: transform A into an upper triangular matrix
        //For each row:
        //1. Search for the maximum element in current column
        //2. Swap maximum row with current row
        //3. Make all of the element 0 below current row in current column
		
        for (int k = 0; k < (A.n - 1); k++){
			if (U(k, k) == 0){
                // Set these to 0.
                maxPivot = 0;
                maxPivotRow = k;

                // Look for higher pivot in rows that are ahead of row k
				for (int i = k + 1 ; i < A.n; i++){
                    // Check that row we want to swap is not also 0
					if (U(i, k) > maxPivot){
                        maxPivot = U(i, k);
                        maxPivotRow = i;
                    }
				}
                // Row swap
                if (maxPivot != 0){
					U = Swap(U, k, maxPivotRow);
                    bTemp = Swap(bTemp, k, maxPivotRow);
				}
                else{
                    // No higher pivot found
                    // Return error
                    cout << "Possibly unsolvable singular system" << endl;
                }
			}
			for (int i = k + 1 ; i < (A.n); i++){
                beta = U(i, k) / U(k,k);
				for (int j = k; j < A.n; j++){
                    U(i, j) = U(i, j) - (beta * U(k, j));
				}
                bTemp(i, 0) = bTemp(i, 0) - (beta * bTemp(k, 0));
			}
		}
        ////Backward Substitution: solve unknowns in a reverse order
        // Solve for the bottom term
        
        solution((A.n - 1), 0) = bTemp((A.n - 1), 0) / U((A.n - 1), (A.n - 1));

        for (int i = (A.n - 2); i >= 0; i--){
            temp = bTemp(i , 0);
            for (int j = (i + 1); j < A.n; j++){
                temp = temp - (U(i, j) * solution(j, 0));
            }
            solution(i, 0) = temp / U(i, i);
        }
        return solution;
    }

    // My additional function to quickly, and simply, access items in my array.
    // parameter (i) represents the row that the item is in.
    // parameter (j) represents the column

    // HW5 - combo assignment
    // Least square linear solver
    Matrix LeastSquareSolve(const Matrix& A, const Matrix& b, Matrix& solution){
        Matrix a = A;
        Matrix A_t = a.Transpose();
        Matrix bTemp = b;
        
        Solve((A_t * a), (A_t * bTemp), solution);

        return solution;
    }

    // Least square linear solver
    Matrix LeastSquareIterativeSolver(Matrix& A, const Matrix& b, Matrix& solution, const int iter){
        Matrix a = A;
        Matrix A_t = a.Transpose();
        Matrix bTemp = b;
        vector<double> plotData; 

        // Obtain a direct solution
        Matrix directSolution(solution.m, solution.n);
        LeastSquareSolve(A, b, directSolution);

        double mu = 1/( SquareSum(A) );
        
        for (int i = 0; i < solution.m * solution.n; i++) {
            solution.data[i] = 0.;
        }
        

        for(int i = 0; i < iter; i++){
    
            solution = solution - (A_t * mu)*((a * solution) - b);
            plotData.push_back(SquareSum(solution - directSolution));  
        }

        Plot_To_File("../plot.txt", plotData);
        

        return solution;
    }

    double SquareSum(const Matrix& A){
        double sum = 0;

        for(int i = 0; i < A.m * A.n; i++){
            sum += (A.data[i] * A.data[i]);
        }

        return sum;
    }

    int Sum(const Matrix& A){
        int sum = 0;

        for(int i = 0; i < A.m * A.n; i++){
            sum += (A.data[i]);
        }

        return sum;
    }

    void Plot_To_File(string file_name, vector<double> Pdata)
    {
        ofstream out(file_name.c_str());
        if(!out){cout<<"Cannot open file "<<file_name<<endl;return;}
        out<<"----Initial Value of ||x(k) - x(hat)----"<<endl;
        for(int i = 0; i < Pdata.size(); i++){
            out<<Pdata[i]<<endl;
        }
        out<<"----Final Value of ||x(k) - x(hat)----"<<endl;
        cout<<"Finish writting to file "<<file_name<<endl;
    }

    void File(string file_name, int h)
    {
        ofstream out(file_name.c_str());
        if(!out){cout<<"Cannot open file "<<file_name<<endl;return;}
        out<<h<<endl;
        cout<<"Finish writting to file "<<file_name<<endl;
    }

    // Combo Assignment 2
    void QRfactorization(Matrix& A, Matrix& Q, Matrix& R){

        //Represent A as a colection of columns
        vector<Matrix> A_columns;


        for(int c = 0; c < A.n; c++){
            Matrix temp(A.m, 1);
            for(int r = 0; r < A.m; r++){
                temp(r, 0) = A(r, c);
            }
            A_columns.push_back(temp);
        }

        // Start Gram Schmit Process
        vector<Matrix> Q_columns;
        
        //for(int c = 0; c < A.m; c++){
        //    Matrix temp(A.m, 1);
        //    A_columns.push_back(temp);
        //}

        for(int i = 0; i < A.n; i++){
            Matrix u(A.m, 1);

            u = A_columns[i];

            for(int j = i; j > 0; j--){
                u = u - Q_columns[j - 1] * (Q_columns[j - 1].Transpose() * A_columns[i]).data[0];
                
                R(j - 1, i) = (A_columns[i].Transpose() * Q_columns[j - 1]).data[0];
                
            }

            if(sqrt(SquareSum(u)) == 0){ return;}
            Q_columns.push_back(u * (1/sqrt(SquareSum(u))));

            
            R(i, i) = sqrt(SquareSum(u));

        }

        
        // Put the columns of Q back into Q
        for(int c = 0; c < A.n; c++){
            for(int r = 0; r < A.m; r++){
                Q(r, c) = Q_columns[c](r, 0);
            }
        }
   
    } 
    

    
};
double Random_Number_Generator()
{
	double rand_n=(double)(rand()%20000-10000)/(double)10000;
	
	return rand_n;
}

void Write_To_File(string file_name, Matrix& solution, Matrix& residual, string message)
{
	ofstream out(file_name.c_str());
	if(!out){cout<<"Cannot open file "<<file_name<<endl;return;}
	out<<message<<endl;
	out<<"Least Square Solution:"<<endl;
	out<<solution<<endl;
	out<<"Residual:"<<endl;
	out<<residual<<endl<<endl;

	cout<<"Finish writting to file "<<file_name<<endl;
}
// Random Matrix generator
Matrix RandomMatrix(int _m, int _n){
    Matrix R(_m, _n);

    for (int i = 0; i < _m * _n; i++) {
        R.data[i] = Random_Number_Generator();
    }
    return R;
}

Matrix IdentityMatrix(int m, int n)
{
    assert(m == n);
    Matrix C(m, n);

    for (int i = 0; i < n; i++) {
        C.data[(i * (n + 1))] = 1;
    }
    return C;
}

void MultiplicationTime(Matrix& A, Matrix& x, Matrix& b)
{
	clock_t start = clock();
	//Do something
	A * x = b;
	clock_t end = clock();
	cout << "Time cost:" << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
}

Matrix SlowMul(const Matrix& A, const Matrix& B)
    {
    assert(A.n == B.m);

    Matrix C(A.m, B.n);

    // loop through rows
    for (int row = 0; row < A.m; row++) {

        // loop through columns
        for (int column = 0; column < B.n; column++) {

            // initialize data to 0
            C.data[(row * B.n) + (column)] = 0;
            // get dot product for specified row and column
            for (int _n = 0; _n < A.n; _n++) {

                C.data[(row * B.n) + (column)] += (A.data[_n + (row * A.n)] * B.data[(_n * B.n) + (column)]);
            }
        }
    }

    return C;
}

void SlowMultiplicationTime(Matrix& A, Matrix& x, Matrix& b)
{
	clock_t start = clock();
	//Do something
	b = SlowMul(A, x);
	clock_t end = clock();
	cout << "Time cost:" << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
}





#endif
