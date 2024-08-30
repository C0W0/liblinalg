//
// Created by Terry on 2024-08-21.
//

#include "Matrix.h"
using namespace linalg;

int main() {
	std::vector<std::vector<int>> v{{1, 2, 3}, {4, 5, 6}};
	Matrix<int> mat(v);
	std::cout << mat << std::endl;
	Matrix<int> matb(v);
	std::vector<std::vector<int>> v2{{1, 2}, {1, 3}, {1, 1}};
	auto result = mat*v2*mat;
	return 1;
}