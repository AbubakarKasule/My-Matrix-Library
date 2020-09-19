#ifndef __MyToolset_h__
#define __MyToolset_h__
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

void Write_To_File_Sample()
{
	cout<<"Write to file sample"<<endl;
	string file_name="test.txt";
	ofstream out(file_name);
	if(!out){cout<<"Cannot open file "<<file_name<<endl;return;}
	out<<1<<endl;
	out<<2<<endl;
	out<<3<<endl;
	cout<<"Finish writting to file "<<file_name<<endl;
}

void Random_Number_Sample()
{
	cout<<"Generate random numbers between -1 to 1"<<endl;
	for(int i=0;i<10;i++){
		double rand_n=(double)(rand()%20000-10000)/(double)10000;
		cout<<"rand "<<rand_n<<endl;
	}
	cout<<"Finish generating random numbers"<<endl;
}

double Random_Number_Generator()
{
	double rand_n=(double)(rand()%20000-10000)/(double)10000;
	
	return rand_n;
}

void Write_To_File(string file_name, Matrix solution, Matrix residual, string message)
{
	ofstream out(file_name);
	if(!out){cout<<"Cannot open file "<<file_name<<endl;return;}
	out<<message<<endl;
	out<<"Least Square Solution:"<<endl;
	out<<solution<<endl;
	out<<"Residual:"<<endl;
	out<<residual<<endl<<endl;

	cout<<"Finish writting to file "<<file_name<<endl;
}
#endif