//////////////////////////////////////////////////////////////////////////
////Dartmouth CS70.01 final project
////http://www.dartmouth.edu/~boolzhu/cosc70.01.html
////Linear algebra and vector math library
//////////////////////////////////////////////////////////////////////////

#ifndef __MyNeuralNetwork_h__
#define __MyNeuralNetwork_h__
#include "MyMatrix.h"
#include <utility>
#include <fstream>
#include <algorithm>

using namespace std;

////Task 1: linear layer
class LinearLayer
{
private:
    Matrix stored_input;     //// Here we should store the input matrix A for Backward
public:
    Matrix weight;
    Matrix weight_grad;     //// record the gradient of the weight

    ////linear layer constructor
    explicit LinearLayer(const int _m=1,const int _n=1):
            weight(Matrix(_m,_n)),weight_grad(Matrix(_m,_n))
    {
        /* _m is the input hidden size and _n is the output hidden size
         * "Kaiming initialization" is important for neural network to converge. The NN will not converge without it!
         */
        for (auto &item : this->weight.data) {
            item=(double)(rand()%20000-10000)/(double)10000*sqrt(6.0/(double)(this->weight.m)); //// Kaiming initialization
        }
    }

    Matrix Forward(Matrix& input)
    {
        /* input.m is batch size and input.n is the #features.
         * 1) Store theinput in stored_data for Backward.
         * 2) Return input * weight.
         */
        assert(input.n==this->weight.m);

        //// Code start ////
		this->stored_input = input;
		Matrix output = input * this->weight;
		return output;
        //// Code end   ////
    }

    ////BE CAREFUL! THIS IS THE MOST CONFUSING FUNCTION. YOU SHOULD READ THE OVERLEAF CAREFULLY BEFORE DIVING INTO THIS!
    Matrix Backward(Matrix& output_grad)
    {
        /* output_grad.m is batch size and output_grad.n is the # output features (this->weight.n)
         * 1) Calculate the gradient of the output (the result of the Forward method) w.r.t. the **weight** and store the product of the gradient and output_grad in weight_grad
         * 2) Calculate the gradient of the output (the result of the Forward method) w.r.t. the **input** and return the product of the gradient and output_grad
         */
        assert(output_grad.n==this->weight.n);

        //// Code start ////
		assert(output_grad.n == this->weight.n);
		this->weight_grad = this->stored_input.Transpose() * output_grad;
		return output_grad * this->weight.Transpose();
        //// Code end   ////
    }
};

////Task 2: non-linear activation
class ReLU
{
private:
    Matrix stored_input;   //// Here we should store the input matrix A for Backward
public:

    ////ReLU layer constructor
    ReLU()=default;

    Matrix Forward(const Matrix& input)
    {
        /*
         *  input_data.m is batch size and input.n is the #features.
         *  This method returns the relu result for each element.
         *  TODO: 1) Go though each element in input and perform relu=max(0,x)
         *  TODO: 2) Store the input in this->stored_data for Backward.
         */

        //// Code start ////
        this->stored_input = input;
        for(int i = 0; i < input.m * input.n; i++){   // loop through all elements
            this->stored_input.data[i] = max((double)0, input.data[i]);
        }
        return stored_input;
        //// Code end   ////
    }

    Matrix Backward(const Matrix& output_grad)
    {
        /*  grad(relu)=1 if relu(x)=x
         *  grad(relu)=0 if relu(x)=0
         *  TODO: returns the gradient of the input data
         *  ATTENTION: Do not forget to multiply the grad(relu) with the output_grad
         */

        //// Code start ////
        Matrix temp = output_grad;

        for(int i = 0; i < output_grad.m * output_grad.n; i++){   // loop through all elements
            if(output_grad.data[i] <= 0){
                temp.data[i] = 0;
            }
        }

		return temp * output_grad;
        //// Code end   ////
    }
};

////Task 3: Loss function
class MSELoss
{
private:
    Matrix stored_data;
public:

    ////cross entropy loss constructor
    MSELoss()= default;

    ////return the mse loss mean(y_j-y_pred_i)^2
    double Forward(Matrix& pred,const Matrix& truth)
    {
        /*  TODO: 1) return MSE(X_pred, X_truth) = ||X_pred-X_truth||^2 / n
         *  TODO: 2) store the difference in this->stored_data for Backward.
         */

        //// Code start ////
        stored_data = pred - truth;
        return pred.SquareSum(stored_data) * (1/pred.m);
        //// Code end   ////
    }

    ////return the gradient of the input data
    Matrix Backward()
    {
        /* TODO 1) return the gradient of the MSE loss: grad(MSE) = 2(X_pred-X_truth) / n
         */
        //// Code start ////
        return stored_data * (2/stored_data.m);
        //// Code end   ////
    }
};

////Task 4: Network architecture
class Network
{
public:
    int n_layers=0;
    vector<LinearLayer> linear_layers;
    vector<ReLU> activation_layers;

    ////MNISTNetwork constructor
    Network(const vector<pair<int, int>>& feature_sizes)
    {
        assert(feature_sizes.size()!=0);
        for (int i=0;i<feature_sizes.size()-1;i++) {assert(feature_sizes[i].second==feature_sizes[i+1].first);}
        /*  TODO: 1) Initialize the array for the linear layers with the feature size specified in the vector feature_sizes.
         * 					 In each pair (in_size, out_size), the in_size is the feature size of the previous layer and the out_size is the feature size of the output (that goes to the next layer)
         * 					 In the linear layer, the weight should have the shape (in_size, out_size).
         *  TODO: 2) Initialize the array for the non-linear layers, the number of which should be feature_size.size()-1.
         *
         *  For example, if feature_size={{256, 128}, {128, 64}, {64, 32}},
       * 							 then there are three linear layers whose weights are with shapes (256, 128), (128, 64), (64, 32),
       * 		   				 and there are two non-linear layers.
         *  Attention: There should be one non-linear layer between two linear layers
         *  					 However, if there is only one linear layer, there should be no non-linear layer at all.
         * 						 The output feature size of the linear layer i should always equal to the input feature size of the linear layer i+1.
       */

        //// Code start ////

        //// Code end   ////
    }

    Matrix Forward(const Matrix& input)
    {
        /* Propagate the input from the first layer to the last layer (before the loss function) by going through the forward functions of all the layers in linear_layers and activation_layers
         * For example, for a network with k linear layers and k-1 activation layers, the data flow is:
         * linear[0] -> activation[0] -> linear[1] ->activation[1] -> ... -> linear[k-2] -> activation[k-2] -> linear[k-1]
         * TODO: 1) propagate the input data throught the network.
         */

        //// Code start ////

        //// Code end   ////
    }

    ////return the gradient of the input data
    Matrix Backward(const Matrix& output_grad)
    {
        /* Propagate the gradient from the last layer to the first layer by going through the backward functions of all the layers in linear_layers and activation_layers
         * TODO: 1) propagate the gradient of the output (the one we got from the Forward method) back throught the network.
         * Notice: We should use the chain rule for the backward.
         * Notice: The order is opposite to the forward.
         */
        //// Code start ////

        //// Code end   ////
    }
};

////Task 5: Matrix slicing
Matrix Matrix_Slice(const Matrix& A, const int start, const int end)
{
    /*  We need to slice the matrix for batch stochastic gradient decent
     *  TODO: 1) Return a matrix with rows of the input A from row 'start' to row 'end-1'.
     */
    //// Code start ////

    //// Code end   ////
}

////Task 6: Regression
class Regressor
{
public:

    Network net;
    MSELoss loss_function=MSELoss();
    Matrix train_data;
    Matrix train_targets;
    Matrix test_data;
    Matrix test_targets;
    double learning_rate=1e-3;
    int max_epoch=200;
    int batch_size=32;

    ////Regressor constructor
    Regressor(vector<pair<int, int>> feature_sizes, double (*unknown_function)(const double)):
            net(Network(feature_sizes)),
            train_data(Matrix(1000,1)), train_targets(Matrix(1000,1)),
            test_data(Matrix(200,1)), test_targets(Matrix(200,1))
    {
        for (int i=0;i<1000;i++) {
            double x=(double)(rand()%20000-10000)/(double)10000;
            double y=unknown_function(x);
            train_data(i, 0)=x;
            train_targets(i, 0)=y;
        }

        for (int i=0;i<200;i++) {
            double x=(double)(rand()%20000-10000)/(double)10000;
            double y=unknown_function(x);
            test_data(i, 0)=x;
            test_targets(i, 0)=y;
        }
    }

    //// Here we train the network using gradient descent
    double Train_One_Epoch()
    {
        double loss=0;
        int n_loop=this->train_data.m/this->batch_size;
        for (int i=1;i<n_loop;i++){
            Matrix batch_data=Matrix_Slice(this->train_data, (i-1)*this->batch_size, i*this->batch_size);
            Matrix batch_targets=Matrix_Slice(this->train_targets, (i-1)*this->batch_size, i*this->batch_size);
            /*  Forward the data to the network.
             *  Forward the result to the loss function.
             *  Backward.
             *  Update the weights with weight gradients.
             *  Do not forget the learning rate!
             */
            //// Code start ////
			Matrix pred = this->net.Forward(batch_data);
			loss += this->loss_function.Forward(pred, batch_targets);
			Matrix pred_grad = this->loss_function.Backward();
			net.Backward(pred_grad);   //// we do not need the gradient for train_data,but just the parameters.
			for (auto& item : this->net.linear_layers) {
				item.weight -= item.weight_grad*this->learning_rate;
			}
            //// Code end   ////
        }
        return loss/(double)n_loop;
    }

    double Test()
    {
        Matrix pred=this->net.Forward(this->test_data);
        double loss=this->loss_function.Forward(pred,this->test_targets);
        return loss;
    }

    void Train()
    {
        for (int i=0;i<this->max_epoch;i++) {
            double train_loss=Train_One_Epoch();
            double test_loss=Test();
            std::cout<<"Epoch: "<<(i+1)<<"/"<<this->max_epoch<<" | Train loss: "<<train_loss<<" | Test loss: "<<test_loss<<std::endl;
        }
    }
};


Matrix One_Hot_Encode(vector<int> labels, int classes=10)
{
    /*  Make the labels one-hot.
     *  For example, if there are 5 classes {0, 1, 2, 3, 4} then
     *  [0, 2, 4] -> [[1, 0, 0, 0, 0],
     * 								[0, 0, 1, 0, 0],
     * 								[0, 0, 0, 0, 1]]
     */
    //// Code start ////

    //// Code end   ////
}


class Classifier
{
public:

    Network net;
    MSELoss loss_function=MSELoss();
    Matrix train_data;  //// The shape should be (m=n_samples,n=28^2)
    vector<int> train_labels;
    Matrix test_data;
    vector<int> test_labels;
    double learning_rate=1e-3;
    int max_epoch=200;
    int batch_size=32;

    ////Classifier constructor
    Classifier(const string& data_dict,const vector<pair<int, int>>& feature_sizes):
            net(Network(feature_sizes)),
            train_data(Matrix(1000,28*28)),test_data(Matrix(200,28*28))
    {
        int tempint;
        ifstream file(data_dict+"/train_data.txt");
        if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
        for(int i=0;i<1000*28*28;i++) if(file >> tempint) train_data.data[i]=(double)tempint/255.0;
        file.close();

        file = ifstream(data_dict+"/train_labels.txt");
        if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
        for(int i=0;i<1000;i++) if(file >> tempint) train_labels.push_back(tempint);
        file.close();

        file = ifstream(data_dict+"/test_data.txt");
        if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
        for(int i=0;i<200*28*28;i++) if(file >> tempint) test_data.data[i]=(double)tempint/255.0;
        file.close();

        file = ifstream(data_dict+"/test_labels.txt");
        if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
        for(int i=0;i<200;i++) if(file >> tempint) test_labels.push_back(tempint);
        file.close();
    }

    double Train_One_Epoch()
    {
        double loss=0;
        int n_loop=this->train_data.m/this->batch_size;
        for (int i=1;i<n_loop;i++){
            Matrix batch_data=Matrix_Slice(this->train_data, (i-1)*this->batch_size, i*this->batch_size);
            auto start=this->train_labels.begin()+(i-1)*this->batch_size;
            vector<int> batch_labels = vector<int>(start, start+this->batch_size);
            /*  Forward the data to the network.
             *  Forward the result to the loss function.
             *  Backward.
             *  Update the weights with weight gradients.
             *  Do not forget the learning rate!
             */
            //// Code start ////

            //// Code end   ////
        }
        return loss/(double)n_loop;
    }

    double Test()
    {
        Matrix score=this->net.Forward(this->test_data);    //// the class with max score is our predicted label
        double accuracy=0;
        for (int i=0;i<score.m;i++) {
            int max_index=0;
            for (int j=0;j<score.n;j++) { if (score(i,j)>score(i,max_index)) {max_index=j;} }
            if (max_index==test_labels[i]) {accuracy+=1;}
        }
        return accuracy/(double)score.m;
    }

    void Train()
    {
        for (int i=0;i<this->max_epoch;i++) {
            double loss=Train_One_Epoch();
            double accuracy=Test();
            std::cout<<"Epoch: "<<(i+1)<<"/"<<this->max_epoch<<" | Train loss: "<<loss<<" | Test Accuracy: "<<accuracy<<std::endl;
        }
    }
};

#endif
