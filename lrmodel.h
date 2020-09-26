#include <bits/c++config.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <algorithm>
using namespace std;
using matrix = vector<vector<double>>;

class lrmodel {
	private:
		size_t N=0;

		matrix x;
		matrix y;

		matrix xx;
		matrix xxx;
		
		matrix xt;

		
		
		
		// methods

				        
        	matrix inverse(matrix A);
        
                
        	matrix matrix_mul(matrix a, matrix b);
        
		matrix transpose(matrix &b);

		void print2d(matrix &v);
	
	
	public:
		lrmodel(matrix  x, matrix y);
		void train();
		matrix beta;
		matrix predict(matrix _x);
		//void save_model(string file_name);


};

lrmodel::lrmodel(matrix ix,matrix iy)
{
	if (ix.size() != iy.size()) {
		throw "SIZE MISMATCH";
	}
	this->N = ix.size();
	this->x=ix;
	this->y=iy;
}

matrix lrmodel::inverse(matrix mat)
{
    // Use Gaussian elimination
    // Using two matrices instead of one augmented
    // to improve performance

    auto height = mat.size();
    auto width = mat[0].size();

    // Create an identity matrix
    matrix result(height, matrix::value_type(width));
    for (auto i = 0;i < width;++i) {
        result[i][i] = 1;
    }
    

    // reduce to Echelon form
    for (auto j = 0;j < width;++j) {
        // partial pivoting
        auto maxRow = j;
        for (auto i = j;i < height;++i) {
            maxRow = mat[i][j]>mat[maxRow][j] ? i : maxRow;
        }
        mat[j].swap(mat[maxRow]);
        result[j].swap(result[maxRow]);

        
        // Reduce row by row
        auto pivot = mat[j][j];
        auto& row1L = mat[j];
        auto& row1R = result[j];
        for (auto i = j + 1;i < height;++i) {
            auto& row2L = mat[i];
            auto& row2R = result[i];
            auto temp = row2L[j];
            for (auto k = 0;k < width;++k) {
                row2L[k] -= temp / pivot*row1L[k];
                row2R[k] -= temp / pivot*row1R[k];
            }
        }
        // Make diaganal elements to 1
        for (auto k = 0;k < width;++k) {
            row1L[k] /= pivot;
            row1R[k] /= pivot;
        }
        
    }

    //back subsitution
    for (auto j = width - 1;;--j) {
        auto& row1L = mat[j];
        auto& row1R = result[j];
        for (auto i = 0;i < j;++i) {
            auto& row2L = mat[i];
            auto& row2R = result[i];
            auto temp = row2L[j];
            for (auto k = 0;k < width;++k) {
                row2L[k] -= temp*row1L[k];
                row2R[k] -= temp*row1R[k];
            }
        }
                
        if (j == 0) break;
    }

    return result;
}

matrix lrmodel::transpose(matrix &b)
{

    matrix trans_vec(b.at(0).size(), vector<double>(b.size()));

    for (size_t i = 0; i < b.size(); i++)
    {
        for (size_t j = 0; j < b.at(i).size(); j++)
        {
            trans_vec.at(j).at(i)=b.at(i).at(j);
        }
    }

    return trans_vec;  
}
matrix lrmodel::matrix_mul(matrix a, matrix b)
{
	unsigned int i1 = a.size(),i2=a[0].size(),i3=b[0].size();
	matrix product(a.size(),vector<double>(b.at(0).size()));
	for (size_t row=0;row<product.size();row++)
		for (size_t col=0;col<product.at(0).size();col++)
			for (size_t inner=0;inner<b.size();inner++) {
				product.at(row).at(col) += a.at(row).at(inner)*b.at(inner).at(col);
			}
	return product;
}


void lrmodel::print2d(matrix &v){
	for ( const auto &row : v )
	{
		for ( const auto &s : row ) cout << s << ' ';
   	cout << endl;
	}
}


void lrmodel::train() {

	
	this -> xt = this->transpose(this->x);

    //matrix xxt = lrmodel::matrix_mul(this->x,this->xt);
	this -> xx = this->inverse(this->matrix_mul(this->xt,this->x));

	this -> xxx = this->matrix_mul(this->xx,this->xt);
	
	this -> beta = this->matrix_mul(this->xxx,this->y);
	
	
}

matrix lrmodel::predict(matrix _x) {
	return this->matrix_mul(_x,this->beta);
}

