//
// Created by Terry on 2024-08-21.
//

#ifndef LINALG_MATRIX_H
#define LINALG_MATRIX_H

#include <vector>
#include <deque>
#include <iostream>
#include <cassert>
#include <memory>
#include <cstring>
#include <variant>

namespace {
	template<class... Ts> struct match : Ts... { using Ts::operator()...; };
	template<class... Ts> match(Ts...) -> match<Ts...>;
}

namespace linalg {
	using std::unique_ptr;
	using std::shared_ptr;
	using std::variant;
	template<typename T>
	using DynArray = std::vector<T>;

	template <typename DT>
	class MatMulResult;
    template <typename DT>
    class MatMulEvalResult;

	// M(height) by N(width) matrix
	template <typename DT>
	class Matrix {
		using DataArray = DynArray<DT>;
		using DataMat = DynArray<DynArray<DT>>;

	public:
		Matrix(): M{0}, N{0} {};

		Matrix(size_t m, size_t n): M{m}, N{n}, data{new DT[m*n]} {
			std::fill(data.get(), data.get()+m*n, 0);
		};

		template<typename DO>
		explicit Matrix(Matrix<DO>& other): M{other.M}, N{other.N}, data{new DT[other.M*other.N]} {
			std::cout << "Copy Constructor Invoked" << std::endl;
			if constexpr (std::is_same_v<DO, DT>) {
				std::memcpy(data.get(), other.data.get(), sizeof(DT)*M*N);
			} else {
				for (int i = 0; i < M; i++) {
					for (int j = 0; j < N; j++) {
						data[i*N+j] = other.data[i*N+j];
					}
				}
			}
		}

		Matrix(Matrix<DT>&& other) noexcept : M{other.M}, N{other.N}, data{std::move(other.data)} {
			std::cout << "Move Constructor Invoked" << std::endl;
		}

		Matrix(const DataMat& inData): M{inData.size()}, N{inData[0].size()}, data{new DT[inData.size() * inData[0].size()]} {
			for (int i = 0; i < M; i++) {
				auto&& row = inData[i];
				assert(row.size() == N);
				std::memcpy(&data[i*N], row.data(), sizeof(DT)*N);
			}
		}

		Matrix(const DataArray& inData): N{1}, M{inData.size()}, data{new DT[inData.size()]} {
			std::memcpy(data.get(), inData.data(), sizeof(DT)*M*N);
		}

		Matrix(const DataArray& inData, size_t m, size_t n): M{m}, N{n}, data{new DT[inData.size()]} {
			assert(m*n == inData.size());
			std::memcpy(data.get(), inData.data(), sizeof(DT)*M*N);
		}

		Matrix(DT* inData, size_t m, size_t n): M{m}, N{n}, data{new DT[m*n]} {
			std::memcpy(data.get(), inData, sizeof(DT)*M*N);
		}

		[[nodiscard]] std::pair<size_t, size_t> dim() const { return {M, N}; };

		void operator *=(const Matrix<DT>& other);
		DT* operator [](size_t i);

        Matrix<DT>& operator =(const Matrix<DT>& other);
        Matrix<DT>& operator =(Matrix<DT>&& other) noexcept;

		void print(std::ostream& iostream) const;
		void print() const;

        DT dataAt(size_t y, size_t x) const;
	private:
		unique_ptr<DT[]> data;
		size_t M; // height
		size_t N; // width
	// friends
		friend class MatMulResult<DT>;
        friend class MatMulEvalResult<DT>;

		friend MatMulResult<DT> operator*(Matrix<DT>& mat1, Matrix<DT>& mat2) {
			assert(mat1.N == mat2.M);
			return MatMulResult<DT>(std::cref(mat1), std::cref(mat2));
		}

		friend MatMulResult<DT> operator*(Matrix<DT>& mat1, Matrix<DT>&& mat2) {
			assert(mat1.N == mat2.M);
			return MatMulResult<DT>(std::cref(mat1), std::make_shared<Matrix<DT>>(std::move(mat2)));
		}

		friend MatMulResult<DT> operator*(Matrix<DT>&& mat1, Matrix<DT>& mat2) {
			assert(mat1.N == mat2.M);
			return MatMulResult<DT>(std::make_shared<Matrix<DT>>(std::move(mat1)), std::cref(mat2));
		}

		friend MatMulResult<DT> operator*(Matrix<DT>&& mat1, Matrix<DT>&& mat2) {
			assert(mat1.N == mat2.M);
			return MatMulResult<DT>(std::make_shared<Matrix<DT>>(std::move(mat1)), std::make_shared<Matrix<DT>>(std::move(mat2)));
		}

		friend std::ostream& operator<<(std::ostream& ostream, const Matrix<DT>& mat) {
			mat.print(ostream);
			return ostream;
		}
	};

	template <typename DT>
	class Vector: public Matrix<DT> {
	public:
		explicit Vector(size_t m = 1): Matrix<DT>(m, 1) {}
		explicit Vector(const DynArray<DT>& inData): Matrix<DT>(inData) {}
	};

	template <typename DT>
	class MatMulResult {
	public:
        using MatRef = variant<std::reference_wrapper<const Matrix<DT>>, shared_ptr<const Matrix<DT>>>;
        using MatArray = std::deque<MatRef>;

		MatMulResult(): M{0}, N{0} {};
		MatMulResult(MatRef m1, MatRef m2) {
			std::visit(match{
				[this](std::reference_wrapper<const Matrix<DT>>& mat) {
					this->M = mat.get().M;
				},
				[this](std::shared_ptr<const Matrix<DT>>& mat) {
					this->M = mat->M;
				}
			}, m1);

			matDims.push_back(M);
			std::visit(match{
					[this](std::reference_wrapper<const Matrix<DT>>& mat) {
						this->N = mat.get().N;
						matDims.push_back(mat.get().M);
					},
					[this](std::shared_ptr<const Matrix<DT>>& mat) {
						this->N = mat->N;
						matDims.push_back(mat->M);
					}
			}, m2);
			matDims.push_back(N);

			matrices.push_back(m1);
			matrices.push_back(m2);
		}

		Matrix<DT> evaluate() const;
		void extend(const MatMulResult<DT>& other);

		void operator *=(const Matrix<DT>& other);
		void operator *=(const MatMulResult<DT>& other);

        operator Matrix<DT>() const;

	private:
		MatArray matrices;
		std::deque<size_t> matDims;
		size_t M = 0; // height
		size_t N = 0; // width

	// friend
		template<typename MR>
		std::enable_if_t<std::is_same_v<std::decay_t<MR>, MatMulResult<DT>>, MatMulResult<DT>>
		friend operator*(MR&& from, Matrix<DT>& mat) {
			assert(from.N == mat.dim().first);
			// if from is lvalue, we duplicated it; if it's rvalue, we move it
			MatMulResult<DT> dup(from);
			dup.matrices.push_back(std::cref(mat));
            dup.matDims.push_back(mat.dim().second);
			dup.N = mat.dim().second;
			return dup;
		}

		template<typename MR>
		std::enable_if_t<std::is_same_v<std::decay_t<MR>, MatMulResult<DT>>, MatMulResult<DT>>
		friend operator*(MR&& from, Matrix<DT>&& mat) {
			assert(from.N == mat.dim().first);
			// if from is lvalue, we duplicated it; if it's rvalue, we move it
			MatMulResult<DT> dup(from);
			dup.matrices.push_back(std::make_shared<Matrix<DT>>(std::move(mat)));
            dup.matDims.push_back(mat.dim().second);
			dup.N = mat.dim().second;
			return dup;
		}

		template<typename MR1, typename MR2>
		std::enable_if_t<std::is_same_v<std::decay_t<MR1>, MatMulResult<DT>> &&
		        std::is_same_v<std::decay_t<MR2>, MatMulResult<DT>>, MatMulResult<DT>>
		friend operator*(MR1&& mr1, MR2&& mr2) {
			assert(mr1.N == mr2.M);
			// if mr1 is lvalue, we duplicated it; if it's rvalue, we move it
			MatMulResult<DT> dup(mr1);
			dup.extend(mr2);
			return dup;
		}

		template<typename MR>
		std::enable_if_t<std::is_same_v<std::decay_t<MR>, MatMulResult<DT>>, MatMulResult<DT>>
		friend operator*(Matrix<DT>& mat, MR&& to) {
			assert(mat.N == to.M);
			// if from is lvalue, we duplicated it; if it's rvalue, we move it
			MatMulResult<DT> dup(to);
			dup.matrices.push_front(std::cref(mat));
            dup.matDims.push_front(mat.dim().first);
			dup.M = mat.dim().first;
			return dup;
		}

		template<typename MR>
		std::enable_if_t<std::is_same_v<std::decay_t<MR>, MatMulResult<DT>>, MatMulResult<DT>>
		friend operator*(Matrix<DT>&& mat, MR&& to) {
			assert(mat.N == to.M);
			// if from is lvalue, we duplicated it; if it's rvalue, we move it
			MatMulResult<DT> dup(to);
			dup.matrices.push_front(std::make_shared<Matrix<DT>>(std::move(mat)));
            dup.matDims.push_front(mat.dim().first);
			dup.M = mat.dim().first;
			return dup;
		}
	};

    template <typename DT>
	class MatMulEvalResult {
	public:
        using EvalProcTable = std::vector<std::vector<MatMulEvalResult<DT>>>;

		MatMulEvalResult(): evaluated{false}, data{std::make_shared<Matrix<DT>>()} {};
		MatMulEvalResult(uint& matIndex, const typename MatMulResult<DT>::MatArray& matrices);
		MatMulEvalResult(const std::pair<uint, uint>& mat1, const std::pair<uint, uint>& mat2,
						 EvalProcTable* table);

		Matrix<DT> matMulPerform(MatMulEvalResult& other);
		void evaluate();

        void set(uint& matIndex, const typename MatMulResult<DT>::MatArray& matrices);
        void set(const std::pair<uint, uint>& mat1, const std::pair<uint, uint>& mat2,
                 EvalProcTable* table);

        Matrix<DT> getData();

	private:
		std::pair<uint, uint> firstMat;
		std::pair<uint, uint> secondMat;
		bool evaluated{};
		typename MatMulResult<DT>::MatRef data;
		EvalProcTable* evalProcTable;
	};

    using MatrixD = Matrix<double>;
	using VecD = Vector<double>;
	using MatrixF = Matrix<float>;
	using VecF = Vector<float>;
	using MatrixC = Matrix<char>;
	using VecC = Vector<char>;
	using MatrixS = Matrix<short>;
	using VecS = Vector<short>;
	using MatrixI = Matrix<int>;
	using VecI = Vector<int>;
	using MatrixL = Matrix<long long>;
	using VecL = Vector<long long>;

} // linalg

#endif //LINALG_MATRIX_H
