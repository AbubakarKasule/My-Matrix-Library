#ifndef __MyGraph_h__
#define __MyGraph_h__
#include <utility>
#include "MyMatrix.h"
#include "MySparseMatrix.h"

class Graph
{
public:
	vector<double> node_values;				////values on nodes
	vector<double> edge_values;				////values on edges
	vector<pair<int,int> > edges;		////edges connecting nodes

	void Add_Node(const double& value)
	{
		node_values.push_back(value);
	}

	void Add_Edge(const int i,const int j,const double value=1.)
	{
		edges.push_back(pair<int,int>(i,j));
		edge_values.push_back(value);
	}

	////display graph in terminal
	friend ostream & operator << (ostream &out,const Graph &graph)
	{
		cout<<"graph node values: "<<graph.node_values.size()<<endl;
		for(int i=0;i<(int)graph.node_values.size();i++){
			cout<<"["<<i<<", "<<graph.node_values[i]<<"] ";
		}
		cout<<endl;

		cout<<"graph edge values: "<<graph.edge_values.size()<<endl;
		for(int i=0;i<(int)graph.edge_values.size();i++){
			cout<<"["<<i<<", "<<graph.edge_values[i]<<"] ";
		}
		cout<<endl;

		cout<<"graph edges: "<<graph.edges.size()<<endl;
		for(int i=0;i<(int)graph.edges.size();i++){
			cout<<"["<<graph.edges[i].first<<", "<<graph.edges[i].second<<"] ";
		}
		cout<<endl;

		return out;
	}

	//////////////////////////////////////////////////////////////////////////
	////Your homework starts

	////HW4 Task 0: build incidence matrix
	void Incidence_Matrix(Matrix& inc_m)
	{
		/*Your implementation starts*/
		// Make sure that incidence matrix is sized correctly
		inc_m.Resize(node_values.size(), edge_values.size());

		// Incase inputed matrix is not empty for some reason
		for (int i = 0; i < inc_m.m * inc_m.n; i++) {
            inc_m.data[i] = 0;
        }

		for (int f = 0; f < edges.size(); f++){
			// Altermatrix at this location
			inc_m(edges[f].first, f) = -1;
			inc_m(edges[f].second, f) = 1;
		}
		/*Your implementation ends*/
	}

	////HW4 Task 1: build adjancency matrix
	void Adjacency_Matrix(/*result*/Matrix& adj_m)
	{
		/*Your implementation starts*/
		// Make sure that adjancency matrix is sized correctly
		adj_m.Resize(node_values.size(), node_values.size());

		// Incase inputed matrix is not empty for some reason
		for (int i = 0; i < adj_m.m * adj_m.n; i++) {
            adj_m.data[i] = 0;
        }

		for (int f = 0; f < edges.size(); f++){
			// Altermatrix at this location
			adj_m(edges[f].first, edges[f].second) = 1;
		}
		/*Your implementation ends*/
	}

	////HW4 Task 3: build the negative Laplacian matrix
	void Laplacian_Matrix(/*result*/Matrix& lap_m)
	{
		/*Your implementation starts*/
		// Make sure that Laplacian matrix is sized correctly
		lap_m.Resize(node_values.size(), node_values.size());
		Matrix inc_m;
		Matrix inc_m_t;
		Incidence_Matrix(inc_m);
		inc_m_t = inc_m.Transpose();

		lap_m = (inc_m * inc_m_t) * -1;
		/*Your implementation ends*/
	}

	////HW4 Task 4: calculate the Dirichlet energy
	double Dirichlet_Energy(const vector<double>& v)
	{
		double de=(double)0;
		/*Your implementation starts*/
		Matrix inc_m;
		Matrix inc_m_t;
		Incidence_Matrix(inc_m);
		inc_m_t = inc_m.Transpose();

		// Turn v into a Matrix for convinnience reasons
		Matrix v_(v.size(), 1);
		for (int i = 0; i < v.size(); i++){ v_(i,0) = v[i];}

		inc_m_t = inc_m_t * v_;

		for (int j = 0; j < inc_m_t.data.size(); j++){
			de += (inc_m_t.data[j] * inc_m_t.data[j]);
		}

		/*Your implementation ends*/

		return de;
	}

	////HW4 Task 5: smooth the node values on the graph by iteratively applying the Laplacian matrix
	void Smooth_Node_Values(const double dt,const int iter_num)
	{
		////copy node_values to local variables
		int m=(int)node_values.size();
		Matrix v(m,1);
		for(int i=0;i<m;i++){
			v(i,0)=node_values[i];
		}
		Matrix v2=v;

		////smooth v
		/*Your implementation starts*/
		// Make the laplatian matrix
		Matrix L(m, m);
		Laplacian_Matrix(L);
		vector<double> tempVector;
		for (int j = 0; j < v.data.size(); j++){
				tempVector.push_back(v.data[j]);
		}
		for (int i = 0; i < iter_num; i++){
			v = v + L * v * dt;
			for (int j = 0; j < v.data.size(); j++){
				tempVector[j] = v.data[j];
			}
			cout<<Dirichlet_Energy(tempVector)<<endl;
		}
		/*Your implementation ends*/

		////copy local variables to node_values
		for(int i=0;i<m;i++){
			node_values[i]=v(i,0);
		}
	}
};

#endif