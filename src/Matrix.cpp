//
// Created by Terry on 2024-08-21.
//

#include "Matrix.h"


namespace {
	using namespace linalg;

	template <typename DT>
	size_t utBuildMatMulProc(
			std::vector<std::vector<size_t>>& table,
            const typename MatMulResult<DT>::MatArray& matrices,
			const std::deque<size_t>& matDims,
			typename MatMulEvalResult<DT>::EvalProcTable& procTable,
			uint32_t start, uint32_t end
	) {
		if (start >= end) {
            if (start == end) {
                procTable[start][end].set(start, matrices);
            }
            return 0;
        }
		if (table[start][end] != 0) {
			return table[start][end];
		}

		size_t ans = SIZE_MAX;
        std::pair<uint32_t, uint32_t> mat1Indices = std::make_pair(start, start);
        std::pair<uint32_t, uint32_t> mat2Indices = std::make_pair(start+1, end);
		for (uint32_t i = start; i < end; i++) {
			size_t val = utBuildMatMulProc<DT>(table, matrices, matDims, procTable, start, i)
                    + utBuildMatMulProc<DT>(table, matrices, matDims, procTable, i+1, end)
                    + matDims[start] * matDims[i+1] * matDims[end+1];
            if (val < ans) {
                mat1Indices.second = i;
                mat2Indices.first = i+1;
                ans = val;
            }
		}

        procTable[start][end].set(mat1Indices, mat2Indices, &procTable);

		table[start][end] = ans;
		return ans;
	}

	template<typename DT>
	Matrix<DT> matMulImpl(const Matrix<DT>& m1, const Matrix<DT>& m2) {
		assert(m1.dim().second == m2.dim().first);
		size_t M = m1.dim().first;
		size_t N = m2.dim().second;
		Matrix<DT> resultData(M, N);

        size_t midDim = m1.dim().second;

		for (size_t y = 0; y < M; y++) {
			for (size_t x = 0; x < N; x++) {
				DT val = 0;
				for (size_t i = 0; i < midDim; i++) {
					val += m1.dataAt(y, i) * m2.dataAt(i, x);
				}
				resultData[y][x] = val;
			}
		}

		return resultData;
	}
}

namespace linalg {
#define LINALG_IMPL(DT) template class MatMulResult<DT>; \
    template class MatMulEvalResult<DT>; \
    template class Matrix<DT>; \
    template class Vector<DT>;

    LINALG_IMPL(double)
    LINALG_IMPL(float)
    LINALG_IMPL(char)
    LINALG_IMPL(short)
    LINALG_IMPL(int)
    LINALG_IMPL(long long)

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
		auto result = MatMulResult<DT>(*this, other).evaluate();
		*this = std::move(result);
	}

	template<typename DT>
	DT* Matrix<DT>::operator[](const size_t i) {
		return &(data.get()[i*N]);
	}

    template<typename DT>
    DT Matrix<DT>::dataAt(size_t y, size_t x) const {
        return data.get()[y*N+x];
    }

    template<typename DT>
    Matrix<DT>& Matrix<DT>::operator=(const Matrix<DT>& other) {
        M = other.M;
        N = other.N;
        unique_ptr<DT[]> newData(new DT[M*N]);
        data.swap(newData);
        std::memcpy(data.get(), other.data.get(), sizeof(DT)*M*N);
        return *this;
    }

    template<typename DT>
    Matrix<DT>& Matrix<DT>::operator=(Matrix<DT>&& other) noexcept {
        M = other.M;
        N = other.N;
        data = std::move(other.data);
        return *this;
    }

	template<typename DT>
	void MatMulResult<DT>::extend(const MatMulResult<DT>& other) {
		assert(N == other.M);
		matrices.insert(matrices.end(), other.matrices.begin(), other.matrices.end());
        matDims.insert(matDims.end(), other.matDims.begin()+1, other.matDims.end());
		N = other.N;
	}

	template<typename DT>
	Matrix<DT> MatMulResult<DT>::evaluate() const {
        int matCount = matrices.size();
		std::vector<std::vector<size_t>> table(matCount, std::vector<size_t>(matCount, 0));
        typename MatMulEvalResult<DT>::EvalProcTable procTable(matCount, std::vector<MatMulEvalResult<DT>>(matCount));
        utBuildMatMulProc<DT>(table, matrices, matDims, procTable, (uint32_t)0, static_cast<uint32_t>(matCount - 1));
        return procTable[0][matCount - 1].getData();
	}

	template<typename DT>
	void MatMulResult<DT>::operator*=(const Matrix<DT>& other) {
		assert(N == other.M);
		matrices.push_back(other);
		N = other.N;
	}

	template<typename DT>
	void MatMulResult<DT>::operator*=(const MatMulResult<DT>& other) {
		assert(N == other.M);
		matrices.insert(matrices.end(), other.matrices.begin(), other.matrices.end());
		N = other.N;
	}

    template<typename DT>
    MatMulResult<DT>::operator Matrix<DT>() const {
        return evaluate();
    }

	template<typename DT>
	MatMulEvalResult<DT>::MatMulEvalResult(
			uint32_t& matIndex, const typename MatMulResult<DT>::MatArray& matrices
	): evaluated{true}, data{matrices[matIndex]} {}

	template<typename DT>
	MatMulEvalResult<DT>::MatMulEvalResult(
			const std::pair<uint32_t, uint32_t>& mat1,
			const std::pair<uint32_t, uint32_t>& mat2,
			MatMulEvalResult::EvalProcTable* table
	): firstMat{mat1}, secondMat{mat2}, evalProcTable{table}, evaluated{false}, data{std::make_shared<Matrix<DT>>()} {}


    template<typename DT>
    void MatMulEvalResult<DT>::set(uint32_t& matIndex, const typename MatMulResult<DT>::MatArray& matrices) {
        data = matrices[matIndex];
        evaluated = true;
    }

    template<typename DT>
    void MatMulEvalResult<DT>::set(
            const std::pair<uint32_t, uint32_t>& mat1,
            const std::pair<uint32_t, uint32_t>& mat2,
            EvalProcTable* table
    ) {
        firstMat = mat1;
        secondMat = mat2;
        evalProcTable = table;
    }

	template<typename DT>
	void MatMulEvalResult<DT>::evaluate() {
        assert(evaluated || !evalProcTable->empty());
		if (!evaluated) {
			MatMulEvalResult<DT>& firstMatEvaled = (*evalProcTable)[firstMat.first][firstMat.second];
			firstMatEvaled.evaluate();
			MatMulEvalResult<DT>& secondMatEvaled = (*evalProcTable)[secondMat.first][secondMat.second];
			secondMatEvaled.evaluate();

			this->data = std::make_shared<Matrix<DT>>(std::move(firstMatEvaled.matMulPerform(secondMatEvaled)));
			evaluated = true;
		}
	}

    template<typename DT>
    Matrix<DT> MatMulEvalResult<DT>::getData() {
        evaluate();
        Matrix<DT> result;
        std::visit(match{
            [&](std::reference_wrapper<const Matrix<DT>>& mat) {
                result = mat.get();
            },
            [&](std::shared_ptr<const Matrix<DT>>& mat) {
                result = *(mat.get());
            }
        }, data);
        return result;
    }

	template<typename DT>
	Matrix<DT> MatMulEvalResult<DT>::matMulPerform(MatMulEvalResult& other) {
		assert(evaluated && other.evaluated);

        const Matrix<DT>* mat1;
        std::visit(match{
            [&](std::reference_wrapper<const Matrix<DT>>& mat) {
                mat1 = &mat.get();
            },
            [&](std::shared_ptr<const Matrix<DT>>& mat) {
                mat1 = mat.get();
            }
        }, data);

        const Matrix<DT>* mat2;
        std::visit(match{
            [&](std::reference_wrapper<const Matrix<DT>>& mat) {
                mat2 = &mat.get();
            },
            [&](std::shared_ptr<const Matrix<DT>>& mat) {
                mat2 = mat.get();
            }
        }, other.data);

		return matMulImpl(*mat1, *mat2);
	}

} // linalg