#include <iostream>
#include "MyMatrix.h"
#include "MySparseMatrix.h"
#include "MyGraph.h"
#include "MyNeuralNetwork.h"
#include "MyLeastSquaresApproximator.h"

/*
void Test_HW1()
{
	////test the sample code
	
	Matrix m1(3,3),m2(3,3);
	
	m1={1.,2.,3.,4.,5.,6.,7.,8.,9.};
	m2={1.,0.,0.,0.,2.,0.,0.,0.,3.};
	cout<<"m1:\n"<<m1<<endl;
	cout<<"m2:\n"<<m2<<endl;

	cout<<"m1+m2:\n"<<(m1+m2)<<endl;

	Matrix m3=m1;
	cout<<"m3=m1, m3:\n"<<m3<<endl;
	
	//////////////////////////////////////////////////////////////////////////
	////start to test your implementation
	////Notice the code will not compile before you implement your corresponding operator
	////so uncomment them one by one if you want to test each function separately
	
	////test subtractions
	
	Matrix m4=-m3;
	cout<<"m4=-m3, m4:\n"<<m4<<endl;
	
	m4=m1-m3;
	cout<<"m4=m1-m3,m4:\n"<<m4<<endl;

	m4-=m1;
	cout<<"m4-=m1, m4:\n"<<m4<<endl;
	

	////test matrix-scalar products
	
	double s=2;
	Matrix m5=m4*s;
	cout<<"m5=m4*s, m5:\n"<<m5<<endl;
	m5*=s;
	cout<<"m5*=s, m5:\n"<<m5<<endl;
	

	////check matrix-matrix multiplication
	
	Matrix v1(3,1);	////column vector
	v1={1.,2.,3.};
	cout<<"column vector v1:\n"<<v1<<endl;

	Matrix v2(1,3);	////row vector
	v2={-3.,-2.,-1.};
	cout<<"row vector v2:\n"<<v2<<endl;

	Matrix v3=v1*v2;
	cout<<"v3=v1*v2, dimension: ["<<v3.m<<", "<<v3.n<<"]"<<endl;
	cout<<"v3 values:\n"<<v3<<endl;

	Matrix v4=v2*v1;
	cout<<"v4=v2*v1, dimension: ["<<v4.m<<", "<<v4.n<<"]"<<endl;
	cout<<"v4 values:\n"<<v4<<endl;

	////test identity, transpose, and block
	
	Matrix m6(3,3);
	cout<<"m6:\n"<<m6.Identity()<<endl;

	Matrix m7(4,2);
	cout<<"m7.Transpose():\n"<<m7.Transpose()<<endl;

	cout<<"m2.Block(0,0,2,2):\n"<<m2.Block(0,0,2,2)<<std::endl;
	
}

void Test_HW2() {
	////test Guassian Elimination
	
	Matrix A1(3, 3);
	Matrix b1(3, 1);
	Matrix x1(3, 1);
	A1 = { 1.,1.,1.,2.,2.,5.,4.,6.,8. };
	b1 = { 1.,2.,3. };
	A1.Solve(A1, b1, x1);
	if (A1*x1 == b1) cout << "Solve A1x=b1:\n " << x1 << endl;
	else cout << "Wrong Answer for A1x=b1" << endl;
	Matrix A2(3, 3);
	Matrix b2(3, 1);
	Matrix x2(3, 1);
	A2 = { 5.,1.,3.,4.,5.,3.,1.,5.,2. };
	b2 = { 3.,6.,-1. };
	A2.Solve(A2, b2, x2);
	if (A2*x2 == b2) cout << "Solve A2x=b2:\n " << x2 << endl;
	else cout << "Wrong Answer for A2x=b2" << endl;
	
	Matrix A3(5, 5);
	Matrix b3(5, 1);
	Matrix x3(5, 1);
	A3 = { 2.,4.,5.,3.,2.,
		4.,8.,3.,4.,3.,
		3.,3.,2.,7.,2.,
		1.,2.,2.,1.,3.,
		3.,4.,2.,5.,7. };
	b3 = { 7.,-4.,-15.,14.,16. };
	cout << A3.Solve(A3, b3, x3) << endl;
	if (A3*x3 == b3) cout << "Solve A3x=b3:\n " << x3 << endl;
	else cout << "Wrong Answer for A3x=b3" << endl;
	
	Matrix A4(1000, 1000);
	Matrix b4(1000, 1);
	Matrix x4(1000, 1);
	for (int i = 1; i < A4.m -1; i++) {
		A4(i, i) = 2.;
		A4(i, i + 1) = -1.;
		A4(i, i - 1) = -1.;
	}
	A4(0, 0) = A4(A4.m-1, A4.n-1) = 2.;
	A4(0, 1) = A4(A4.m - 1, A4.n - 2) = -1;
	for (int i = 0; i < b4.m; i++) { b4(i, 0) = (double)i/(double)(b4.m*b4.m); }
	A4.Solve(A4, b4, x4);
	if (A4*x4 == b4) cout << "Solve A4x=b4:\n " << x4 << endl;
	else cout << "Wrong Answer for A4x=b4" << endl;
	

}

void Test_HW3()
{
	std::cout<<"Test sparse matrix"<<std::endl;
	SparseMatrix mtx(5,5);
	vector<tuple<int,int,double> > elements;
	elements.push_back(make_tuple<int,int,double>(0,0,7));
	elements.push_back(make_tuple<int,int,double>(0,1,5));
	elements.push_back(make_tuple<int,int,double>(1,0,1));
	elements.push_back(make_tuple<int,int,double>(1,2,3));
	elements.push_back(make_tuple<int,int,double>(2,3,5));
	elements.push_back(make_tuple<int,int,double>(2,4,4));
	elements.push_back(make_tuple<int,int,double>(3,3,1));
	elements.push_back(make_tuple<int,int,double>(4,1,7));
	elements.push_back(make_tuple<int,int,double>(4,4,3));
	mtx=elements;

	cout<<"sparse matrix:\n"<<mtx<<endl;

	Matrix v(5,1);v={1,2,3,4,5};
	Matrix prod(5,1);
	prod=mtx*v;
	cout<<"sparse matrix-vector multiplication:\n";
	cout<<prod<<endl;
}

void Test_HW4()
{
	std::cout<<"Test graph matrix"<<std::endl;
	Graph g;
	g.Add_Node(0.);
	g.Add_Node(1.);
	g.Add_Node(2.);
	g.Add_Node(3.);
	g.Add_Node(4.);
	g.Add_Node(5.);

	g.Add_Edge(0,1);
	g.Add_Edge(1,2);
	g.Add_Edge(1,3);
	g.Add_Edge(2,3);
	g.Add_Edge(2,4);
	g.Add_Edge(3,4);
	g.Add_Edge(4,5);
	
	Matrix adj_m;g.Adjacency_Matrix(adj_m);
	Matrix inc_m;g.Incidence_Matrix(inc_m);
	Matrix lap_m;g.Laplacian_Matrix(lap_m);
	double energy=g.Dirichlet_Energy(g.node_values);
	
	cout<<g<<endl;
	cout<<"Adjacency matrix\n"<<adj_m<<endl;
	cout<<"Incidency matrix\n"<<inc_m<<endl;
	cout<<"Laplacian matrix\n"<<lap_m<<endl;
	cout << "Dirichlet energy before smoothing: "<<energy<<endl;

	g.Smooth_Node_Values(.1,10);
	energy=g.Dirichlet_Energy(g.node_values);
	cout<<"Dirichlet energy after smoothing: "<<energy<<endl;
	

}


void Test_HW5()
{

	// Write_To_File_Sample();
	// Random_Number_Sample();

	Matrix A = RandomMatrix(30, 10);
	Matrix b = RandomMatrix(30, 1);

	Matrix solved(10, 1);
	solved.LeastSquareSolve(A, b, solved);

	Matrix c = (A * solved) - b;
	Matrix c_t = c.Transpose();

	Matrix residual = c_t * c;

	Write_To_File("../question2", solved, residual, "Initial Least square solution and residual");

	Matrix d1 = RandomMatrix(10, 1);
	Matrix d2 = RandomMatrix(10, 1);
	Matrix d3 = RandomMatrix(10, 1);

	c = (A * (solved + d1)) - b;
	c_t = c.Transpose();
	residual = c_t * c;

	Write_To_File("../question2_d1", solved, residual, "Least square solution and residual after adding d1");

	c = (A * (solved + d2)) - b;
	c_t = c.Transpose();
	residual = c_t * c;

	Write_To_File("../question2_d2", solved, residual, "Least square solution and residual after adding d2");

	c = (A * (solved + d3)) - b;
	c_t = c.Transpose();
	residual = c_t * c;

	Write_To_File("../question2_d3", solved, residual, "Least square solution and residual after adding d3");


	// Iterative Solver
	Matrix a_A = RandomMatrix(20, 10);
	Matrix a_b = RandomMatrix(20, 1);

	Matrix a_solved(10, 1);
	cout<<a_solved.LeastSquareIterativeSolver(a_A, a_b, a_solved, 500)<<endl;



}


void Test_HW6(){
	
	// QR factorization

	//// #1
	Matrix temp;
	Matrix _A = RandomMatrix(20, 10);
	Matrix _b = RandomMatrix(20, 1);
	Matrix Q(20, 10);
	Matrix R(10, 10);
	Matrix x(10, 1);
	

	temp.QRfactorization(_A, Q, R);

	temp = Q.Transpose() * _b;

	Matrix x_h(10, 1);

	temp.LeastSquareSolve(_A, _b, x);
	temp.Solve(R, temp, x_h);

	cout<<"X hat: "<<x_h<<endl;
	cout<<"X :"<<x<<endl;


	//// #2
	if(_A * x_h == Q * Q.Transpose() * _b){
		cout<<"True"<<endl;
	}
	
	// high-performance linear algebra
	// oPTIMIZATION TESTING
	Matrix A = RandomMatrix(10, 10);
	Matrix b = RandomMatrix(10, 10);
	Matrix c;

	MultiplicationTime(A, b, c);
	SlowMultiplicationTime(A, b, c);

	A = RandomMatrix(100, 100);
	b = RandomMatrix(100, 100);

	MultiplicationTime(A, b, c);
	SlowMultiplicationTime(A, b, c);

	A = RandomMatrix(1000, 1000);
	b = RandomMatrix(1000, 1000);

	
	MultiplicationTime(A, b, c);
	SlowMultiplicationTime(A, b, c);

	
	A = RandomMatrix(10000, 10000);
	b = RandomMatrix(10000, 10000);

	MultiplicationTime(A, b, c);
	SlowMultiplicationTime(A, b, c); 

}

double data_function(const double x)  {return pow(x,3)+pow(x,2)+1.;}


void Test_Final()
{
    vector<pair<int, int>> regressor_feature_sizes={{1, 16}, {16, 16}, {16, 16}, {16, 1}};
    Regressor reg(regressor_feature_sizes,&data_function);
    reg.Train();
    vector<pair<int, int>> classifier_feature_sizes={{28*28, 256}, {256, 256}, {256, 256}, {256, 10}};
    Classifier cls("../Release/MNIST_Sub", classifier_feature_sizes);
    cls.Train();
}
*/

void test(){
	double margin = 0.001;
	Matrix pred, truth;

	cout<<"MSE for function approximator: "<<functionApproximatorTest()<<endl;

	cout<<"MSE for image classifier: "<<Classifier()<<endl;

	cout<<"The Classifier predicted correctly "<<QRClassifierPercentage(pred, truth, 1000, 200)<<"% of the time"<<endl;

	cout<<"The Function approximator predicted correctly "<<functionApproximatorTestPercentage(0.001)<<"% of the time within a margin of "<<margin<<endl;

	cout<<"The QR Classifier predicted correctly "<<QRClassifierPercentage(pred, truth, 1000, 200)<<"% of the time"<<endl;
}


int main()
{
	std::cout<<"Hello CS70!"<<std::endl;

	// Test_HW1();
	// Test_HW2();
	// Test_HW3();
	// Test_HW4();
	// Test_HW5();
	// Test_HW6();
	test();

	system("PAUSE");
}
