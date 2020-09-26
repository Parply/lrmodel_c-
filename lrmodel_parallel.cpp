#include "lrmodel_parallel.h"
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <stdexcept>
using matrix=vector<vector<double>>;
void print2d(matrix &v){
	for ( const auto &row : v )
	{
		for ( const auto &s : row ) cout << s << ' ';
   	cout << endl;
	}
}


vector<matrix> readCSV(string file_name){
	 
	ifstream file(file_name);
	if(!file.is_open()) throw runtime_error("Could not open file");
	vector<double> row;
	char delimiter=',';
	string line,word;
	double val;
	size_t lines_count=0,i=0;
	
	while (getline(file, line)){
        	++lines_count;
	}
	lines_count-=1;
	matrix tt(lines_count,vector<double>()); 
	file.clear();
	file.seekg(0,ios::beg);
	getline(file,line);
	
	while (getline(file,line)){
		
		//vec.clear();
		stringstream ss(line);
		while (ss>>val){
			tt.at(i).push_back(val);
			if(ss.peek() == delimiter) ss.ignore();
		}
		
		i++;
		
	}
	file.close();
	
	matrix x(lines_count,vector<double>()),y(lines_count,vector<double>());
	#pragma omp parallel for schedule(static)
	for (size_t row=0;row<tt.size();row++){
		for(size_t i=0;i<tt.at(row).size()-1;i++){
			x.at(row).push_back(tt.at(row).at(i));
		}
		y.at(row).push_back(tt.at(row).at(tt.at(row).size()-1));
	}
	
	vector<matrix> out;
	out.push_back(x);
	out.push_back(y);
	return out;
	
}


int main() {

	vector<matrix> out=readCSV("creditcard.csv");
	matrix x = out.at(0);
	matrix y = out.at(1);
	lrmodel lr(x,y);
	lr.train();
	matrix pred=lr.predict(x);
	/*
	cout << "Pred: " << endl;
	for (const auto row: pred){
		for (const auto col:row){
			cout << col << " ";
		}
		cout << endl;
	}*/
	return 0;
}
