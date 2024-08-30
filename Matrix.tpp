//
// Created by Terry on 2024-08-21.
//

#include "Matrix.h"

namespace {
	using namespace linalg;

	template <typename DT>
	size_t utBuildMatMulProc(
			std::vector<std::vector<size_t>>& table,
			const std::deque<size_t>& matDims,
			typename MatMulEvalResult<DT>::EvalProcTable& procTable,
			uint start, uint end
	) {
		if (start >= end) return 0;
		if (table[start][end] != 0) {
			return table[start][end];
		}

		size_t ans = SIZE_MAX;
		for (uint i = start; i < end; i++) {
			size_t val = utBuildMatMulProc(table, matDims, procTable, start, i) + utBuildMatMulProc(table, matDims, procTable, i+1, end) +
						 matDims[start] * matDims[i+1] * matDims[end+1];
			ans = std::min(ans, val);
		}

		table[start][end] = ans;
		return ans;
	}

	template<typename DT>
	Matrix<DT> matMulImpl(const Matrix<DT>& m1, const Matrix<DT>& m2) {
		assert(m1.dim().second == m2.dim().first);
		size_t M = m1.dim().first;
		size_t N = m2.dim().second;
		Matrix<DT> resultData(M, N);

		for (size_t y = 0; y < M; y++) {
			for (size_t x = 0; x < m2.N; x++) {
				DT val = 0;
				for (size_t i = 0; i < N; i++) {
					val += m1[y][i] * m2[i][x];
				}
				resultData[y][x] = val;
			}
		}

		return resultData;
	}
}

namespace linalg {
	template<typename DT>
	void Matrix<DT>::print(std::ostream& iostream) const {
		iostream << "[";
		for (int i = 0; i < M; i++) {
			iostream << "[ ";
			for (int j = 0; j < N; j++) {
				iostream << data[i*N+j] << ", ";
			}
			iostream << ']';
			if (i == M-1)
				iostream << ']';
			else
				iostream << std::endl;
		}
	}

	template<typename DT>
	void Matrix<DT>::print() const {
		this->print(std::cout);
		std::cout << std::endl;
	}

	template<typename DT>
	void Matrix<DT>::operator*=(const Matrix<DT>& other) {
		assert(N == other.M);
		auto result = MatMulResult<DT>(this, &other).evaluate();
		this = std::move(result);
	}

	template<typename DT>
	DT* Matrix<DT>::operator[](const size_t i) {
		return &(data.get()[i*N]);
	}

	template<typename DT>
	void MatMulResult<DT>::extend(const MatMulResult<DT>& other) {
		assert(N == other.M);
		matrices.insert(matrices.end(), other.matrices.begin(), other.matrices.end());
		N = other.N;
	}

	template<typename DT>
	Matrix<DT> MatMulResult<DT>::evaluate() {
		std::vector<std::vector<DT>> table;
		return Matrix<DT>();
	}

	template<typename DT>
	void MatMulResult<DT>::operator*=(const Matrix<DT>& other) {
		assert(N == other.M);
		matrices.push_back(&other);
		N = other.N;
	}

	template<typename DT>
	void MatMulResult<DT>::operator*=(const MatMulResult<DT>& other) {
		assert(N == other.M);
		matrices.insert(matrices.end(), other.matrices.begin(), other.matrices.end());
		N = other.N;
	}

	template<typename DT>
	MatMulEvalResult<DT>::MatMulEvalResult(
			uint& matIndex, const typename MatMulResult<DT>::MatArray& matrices
	): evaluated{true}, data{matrices[matIndex]} {}

	template<typename DT>
	MatMulEvalResult<DT>::MatMulEvalResult(
			const std::pair<uint, uint>& mat1,
			const std::pair<uint, uint>& mat2,
			MatMulEvalResult::EvalProcTable& table
	): firstMat{mat1}, secondMat{mat2}, evalProcTable{table}, evaluated{false} {}

	template<typename DT>
	void MatMulEvalResult<DT>::evaluate() {
		if (!evaluated) {
			MatMulEvalResult<DT>& firstMatEvaled = evalProcTable[firstMat.first][firstMat.second];
			firstMatEvaled.evaluate();
			MatMulEvalResult<DT>& secondMatEvaled = evalProcTable[secondMat.first][secondMat.second];
			secondMatEvaled.evaluate();

			this->data = std::make_shared<Matrix<DT>>(std::move(firstMatEvaled.matMulPerform(secondMatEvaled)));
			evaluated = true;
		}
	}

	template<typename DT>
	Matrix<DT> MatMulEvalResult<DT>::matMulPerform(const MatMulEvalResult& other) {
		assert(evaluated && other.evaluated);
		return matMulImpl(this->data, other.data);
	}

} // linalg