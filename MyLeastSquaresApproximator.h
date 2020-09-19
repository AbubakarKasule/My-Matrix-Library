#include "MyMatrix.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>

void getAllDataAndLabels(Matrix& train_data, Matrix& train_labels, Matrix& test_data, Matrix& test_labels, const string& data_dict, int numTrain, int numTest){
    
    int tempint;
    ifstream file("./train_data.txt");
    if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
    for(int i=0;i<numTrain*28*28;i++) if(file >> tempint) train_data.data[i]=(double)tempint/255.0;
    file.close();

    vector<int> train_l;
    file = ifstream("./train_labels.txt");
    if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
    for(int i=0;i<numTrain;i++) if(file >> tempint) train_l.push_back(tempint);
    file.close();
    for(int i = 0; i < train_l.size(); i++){ train_labels.data[i] = train_l[i];}

    file = ifstream("./test_data.txt");
    if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
    for(int i=0;i<numTest*28*28;i++) if(file >> tempint) test_data.data[i]=(double)tempint/255.0;
    file.close();

    vector<int> test_l;
    file = ifstream("./test_labels.txt");
    if (!file.is_open()) std::cout<<"file does not exist"<<std::endl;
    for(int i=0;i<numTest;i++) if(file >> tempint) test_l.push_back(tempint);
    file.close();
    for(int i = 0; i < test_l.size(); i++){ test_labels.data[i] = test_l[i];}


}



Matrix generateData(Matrix& xValues, Matrix& yValues, int numOfDataPoints, int maxPower){
    double temp;
    Matrix newX(numOfDataPoints, maxPower);

    for(int j = 0; j < numOfDataPoints; j++){
        temp = 0.0;
        for(int i = 0; i < maxPower; i++){             // Apply basis functions
            temp = temp + pow(xValues(j, 0), i);
            newX(j, i) = pow(xValues(j, 0), i);
        }
        yValues(j, 0) = temp;
    }

    
 
    return newX; 
}

/*
Matrix functionApproximator(Matrix& data, Matrix& labels){
    Matrix theta(data.n, 10);

    theta.LeastSquareSolve(data, labels, theta);

    return theta;

}
*/

Matrix addBias(Matrix& data){
    Matrix result(data.m, data.n + 1);           // Add an extra column for the bias

    for(int i = 0; i < result.m; i++){           // Set extra columns
        result(i, 0) = 1;
    }

    for(int i = 1; i < result.n; i++){            
        for(int j = 0; j < data.m; j++){
            result(j, i) = data(j, i - 1);
        }
    }
 
    return result; 
}

Matrix boolerizeLabels(Matrix& label, int numberOfLabels){
    // Turn them into bools

    Matrix result(label.m, numberOfLabels);

    for(int i = 0; i < label.m; i++){
        for(int j = 0; j < numberOfLabels; j++){
            if(label(i, 0) == j){
                result(i, j) = 1.0;
            }
            else{
                result(i, j) = -1.0;
            }
        }
    }

    return result;
}

double MSE(Matrix& X_pred, Matrix& X_truth){
    Matrix r;
    r = X_truth - X_pred;
    double MSE = sqrt(X_pred.SquareSum(r))/X_pred.m; 
    return MSE;
}

void columnize(Matrix& col, Matrix& a, int i){
    for(int j = 0; j < a.m; j++){
        col(j,0) = a(j,i); //store ith column of train labels
    }
}

Matrix functionApproximator(Matrix& train_data, Matrix& train_labels){
    Matrix theta(train_data.n, train_labels.n);

    // find theta
    theta.LeastSquareSolve(train_data, train_labels, theta);
    
    return theta;
}

void saltAndPepperMatrix(Matrix& A){
    double salt; 

    for(int i = 0; i < A.m; i++){
        for(int j = 0; j < A.n; j++){
            salt = ((double)(rand()%10000)/(double)10000) * 0.0011;
            A(i, j) = A(i, j) + salt;
        }
    }
    
}

double functionApproximatorTest(){
     // Define the matrices
    Matrix train_data = RandomMatrix(1000, 1);
    Matrix train_labels(1000, 1);
    Matrix test_data = RandomMatrix(200, 1);
    Matrix test_labels(200, 1);
    Matrix new_train_data;
    Matrix new_test_data;
    Matrix theta;
    
    // Generate data
    new_train_data = generateData(train_data, train_labels, 1000, 4);
    new_test_data = generateData(test_data, test_labels, 200, 4);

    
    // Approx. the function
    theta = functionApproximator(new_train_data, train_labels);

    Matrix f = new_test_data * theta;

    //cout<<theta.m<<" "<<theta.n<<" | "<<f.m<<" "<<f.n<<" | "<<test_labels.m<<" "<<test_labels.n<<endl;
    

    return MSE(f, test_labels);
    
}

double functionApproximatorTestPercentage(double marginOfError){
     // Define the matrices
    Matrix train_data = RandomMatrix(1000, 1);
    Matrix train_labels(1000, 1);
    Matrix test_data = RandomMatrix(200, 1);
    Matrix test_labels(200, 1);
    Matrix new_train_data;
    Matrix new_test_data;
    Matrix theta;
    
    // Generate data
    new_train_data = generateData(train_data, train_labels, 1000, 4);
    new_test_data = generateData(test_data, test_labels, 200, 4);

    
    // Approx. the function
    theta = functionApproximator(new_train_data, train_labels);

    Matrix prediction = new_test_data * theta;

    double final = 0.0;

     
    double total;
    
    total = 0.0;

    for(int i = 0; i < prediction.m; i++){
        
        if(prediction(i, 0) - test_labels(i, 0) > (-1 * marginOfError) && prediction(i, 0) - test_labels(i, 0) < marginOfError){
            total = total + 1.0;
        }
    }
    
    final = (total/(double)prediction.m) * (double)100;


    return final;
}

double Classifier(){
    
    // Define the matrices
    Matrix train_data(1000, 784);
    Matrix train_labels(1000, 1);
    Matrix test_data(200, 784);
    Matrix test_labels(200, 1);
    Matrix new_train_data;
    Matrix new_test_data;
    Matrix new_train_labels;
    Matrix new_test_labels;


    // Load values
    getAllDataAndLabels(train_data, train_labels, test_data, test_labels, "../Release/MNIST_Sub", 1, 1);
    
     // Salt and pepper
    saltAndPepperMatrix(train_data);
    saltAndPepperMatrix(test_data);

    // Apply basis functions and add bias
    // new_train_data and new_test_data have dimensions (1000, 785) and (200, 785) respectivley
    new_train_data = addBias(train_data);
    new_test_data = addBias(test_data);
    
    
    // bOOLER1Ze
    new_train_labels = boolerizeLabels(train_labels, 10);
    new_test_labels = boolerizeLabels(test_labels, 10);


    // Perform Least squares
    Matrix theta(785, 1);
    Matrix result(new_train_data.n, 10);

    
    // result.LeastSquareSolve(new_train_data, new_train_labels, result);
    

    Matrix label(1000, 1); 
    
    for(int i = 0; i < 10; i++){
        columnize(label,new_train_data, i);

        // cout<<new_train_data.m<<" "<<new_train_data.n<<" | "<<label.m<<" "<<label.n<<" | "<<theta.m<<" "<<theta.n<<endl;

        theta.LeastSquareSolve(new_train_data, label, theta);

        for(int j = 0; j <  theta.m; j++){
            result(j,i) = theta(j,0);
        }
    }
    
    Matrix prediction = new_test_data * result;

    return MSE(prediction, new_test_labels);
}

double ClassifierPercentage(Matrix& pred, Matrix& truth, int numTrain, int numTest){
    
    // Define the matrices
    Matrix train_data(numTrain, 784);
    Matrix train_labels(numTrain, 1);
    Matrix test_data(numTest, 784);
    Matrix test_labels(numTest, 1);
    Matrix new_train_data;
    Matrix new_test_data;
    Matrix new_train_labels;
    Matrix new_test_labels;


    // Load values
    getAllDataAndLabels(train_data, train_labels, test_data, test_labels, "../Release/MNIST_Sub", numTrain, numTest);
    
    // Salt and pepper
    saltAndPepperMatrix(train_data);
    saltAndPepperMatrix(test_data);

    // Apply basis functions and add bias
    // new_train_data and new_test_data have dimensions (1000, 785) and (200, 785) respectivley
    new_train_data = addBias(train_data);
    new_test_data = addBias(test_data);
    
    
    // bOOLER1Ze
    new_train_labels = boolerizeLabels(train_labels, 10);
    new_test_labels = boolerizeLabels(test_labels, 10);


    // Perform Least squares
    Matrix theta(785, 1);
    Matrix result(new_train_data.n, 10);

    
    // result.LeastSquareSolve(new_train_data, new_train_labels, result);

    Matrix label(numTrain, 1); 
    
    for(int i = 0; i < 10; i++){
        columnize(label,new_train_labels, i);

        // cout<<new_train_data.m<<" "<<new_train_data.n<<" | "<<label.m<<" "<<label.n<<" | "<<theta.m<<" "<<theta.n<<endl;

        theta.LeastSquareSolve(new_train_data, label, theta);

        for(int j = 0; j < theta.m; j++){
            result(j,i) = theta(j,0);
        }
    }
    
    Matrix prediction = new_test_data * result;

    double final = 0.0;

     
    double total, m1, m2;

    int i1, i2;

    
    m2 = 0.0;
    total = 0.0;

    for(int i = 0; i < prediction.m; i++){
        m1 = prediction(i, 0);
        for(int j = 0; j < prediction.n; j++){
            if(new_test_labels(i, j) == 1){ i1 = j;}
            if(prediction(i, j) > m1){ 
                i2 = j;
                m1 = prediction(i, j);
            }
        }
        if(i1 == i2){
            total = total + 1.0;
        }
    }
    
    final = (total/(double)prediction.m) * (double)100.0;


    return final;
}

double QRClassifierPercentage(Matrix& pred, Matrix& truth, int numTrain, int numTest){
    
    // Define the matrices
    Matrix train_data(numTrain, 784);
    Matrix train_labels(numTrain, 1);
    Matrix test_data(numTest, 784);
    Matrix test_labels(numTest, 1);
    Matrix new_train_data;
    Matrix new_test_data;
    Matrix new_train_labels;
    Matrix new_test_labels;


    // Load values
    getAllDataAndLabels(train_data, train_labels, test_data, test_labels, "../Release/MNIST_Sub", numTrain, numTest);
    
    // Salt and pepper
    saltAndPepperMatrix(train_data);
    saltAndPepperMatrix(test_data);

    // Apply basis functions and add bias
    // new_train_data and new_test_data have dimensions (1000, 785) and (200, 785) respectivley
    new_train_data = addBias(train_data);
    new_test_data = addBias(test_data);
    
    
    // bOOLER1Ze
    new_train_labels = boolerizeLabels(train_labels, 10);
    new_test_labels = boolerizeLabels(test_labels, 10);


    // Perform Least squares
    Matrix theta(785, 1);
    Matrix result(new_train_data.n, 10);

    
    // result.LeastSquareSolve(new_train_data, new_train_labels, result);

    Matrix label(numTrain, 1); 

    Matrix Q(new_train_data.m, new_train_data.n);
    Matrix R(new_train_data.n, new_train_data.n);

    
    new_train_data.QRfactorization(new_train_data, Q, R);
   
    for(int i = 0; i < 10; i++){
        columnize(label,new_train_labels, i);

        // cout<<new_train_data.m<<" "<<new_train_data.n<<" | "<<label.m<<" "<<label.n<<" | "<<theta.m<<" "<<theta.n<<endl;
        
        theta.Solve(R, (Q.Transpose() * label), theta);

        for(int j = 0; j < theta.m; j++){
            result(j,i) = theta(j,0);
        }
    }
    
    Matrix prediction = new_test_data * result;

    double final = 0.0;

     
    double total, m1, m2;

    int i1, i2;

    
    m2 = 0.0;
    total = 0.0;

    for(int i = 0; i < prediction.m; i++){
        m1 = prediction(i, 0);
        for(int j = 0; j < prediction.n; j++){
            if(new_test_labels(i, j) == 1){ i1 = j;}
            if(prediction(i, j) > m1){ 
                i2 = j;
                m1 = prediction(i, j);
            }
        }
        if(i1 == i2){
            total = total + 1.0;
        }
    }
    
    final = (total/(double)prediction.m) * (double)100.0;


    return final;
}