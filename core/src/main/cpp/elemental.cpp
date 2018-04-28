#include "alchemist.h"

namespace alchemist {

void ThinSVDCommand::run(Worker *self) const {
  auto m = self->matrices[mat]->Height();
  auto n = self->matrices[mat]->Width();
  auto k = std::min(m, n);
  DistMatrix * U = new El::DistMatrix<double, El::VR, El::STAR>(m, k, self->grid);
  DistMatrix * singvals = new El::DistMatrix<double, El::VR, El::STAR>(k, k, self->grid);
  DistMatrix * V = new El::DistMatrix<double, El::VR, El::STAR>(n, k, self->grid);
  El::Zero(*U);
  El::Zero(*V);
  El::Zero(*singvals);

  ENSURE(self->matrices.insert(std::make_pair(Uhandle, std::unique_ptr<DistMatrix>(U))).second);
  ENSURE(self->matrices.insert(std::make_pair(Shandle, std::unique_ptr<DistMatrix>(singvals))).second);
  ENSURE(self->matrices.insert(std::make_pair(Vhandle, std::unique_ptr<DistMatrix>(V))).second);

  DistMatrix * Acopy = new El::DistMatrix<double, El::VR, El::STAR>(m, n, self->grid); // looking at source code for SVD, seems that DistMatrix Acopy(A) might generate copy rather than just copy metadata and risk clobbering
  El::Copy(*self->matrices[mat], *Acopy);
  El::SVD(*Acopy, *U, *singvals, *V);
  std::cerr << format("%s: singvals is %s by %s\n") % self->world.rank() % singvals->Height() % singvals->Width();
  self->world.barrier();
}

void MatrixMulCommand::run(Worker *self) const {
  auto m = self->matrices[inputA]->Height();
  auto n = self->matrices[inputB]->Width();
  self->log->info("Arrived in matrix multiplication code");
  auto A = dynamic_cast<El::DistMatrix<double, El::VR, El::STAR>*>(self->matrices[inputA].get());
  auto B = dynamic_cast<El::DistMatrix<double, El::VR, El::STAR>*>(self->matrices[inputB].get());
  auto C = new El::DistMatrix<double, El::VR, El::STAR>(m, n, self->grid);
  ENSURE(self->matrices.insert(std::make_pair(handle, std::unique_ptr<DistMatrix>(C))).second);
  self->log->info("Starting multiplication");
  //Gemm(1.0, *A, *B, 0.0, *C, self->log);
  El::Gemm(El::NORMAL, El::NORMAL, 1.0, *A, *B, 0.0, *C);
  self->log->info("Done with multiplication");
  self->world.barrier();
}

void TransposeCommand::run(Worker *self) const {
  auto m = self->matrices[origMat]->Height();
  auto n = self->matrices[origMat]->Width();
  DistMatrix * transposeA = new El::DistMatrix<double, El::VR, El::STAR>(n, m, self->grid);
  El::Zero(*transposeA);

  ENSURE(self->matrices.insert(std::make_pair(transposeMat, std::unique_ptr<DistMatrix>(transposeA))).second);

  El::Transpose(*self->matrices[origMat], *transposeA);
  std::cerr << format("%s: finished transpose call\n") % self->world.rank();
  self->world.barrier();
}

void LeastAbsoluteDeviationsCommand::run(Worker * self) const {
  auto A = self->matrices[Amat].get();
  auto b = self->matrices[bvec].get();
  auto n = A->Width();
  auto x = new El::DistMatrix<double, El::VR, El::STAR>(n, 1, self->grid);
  ENSURE(self->matrices.insert(std::make_pair(xvec, std::unique_ptr<DistMatrix>(x))).second);

  El::DistMatrix<double, El::MC, El::MR> Acopy(*A);
  El::DistMatrix<double, El::MC, El::MR> bcopy(*b);
  El::DistMatrix<double, El::MC, El::MR> xcopy(self->grid);
  El::LAV(Acopy, bcopy, xcopy);
  El::Copy(xcopy, *x);
  self->world.barrier();
}

void NormalizeMatInPlaceCommand::run(Worker *self) const {
    auto m = self->matrices[A]->Height();
    auto n = self->matrices[A]->Width();
    auto matA = self->matrices[A].get();
    auto localRows = matA->LockedMatrix();

    auto rowMeans = El::DistMatrix<double, El::STAR, El::STAR>(n, 1, self->grid);
    auto distOnes = El::DistMatrix<double, El::STAR, El::STAR>(m, 1, self->grid);
    El::Matrix<double>localRowMeans;
    El::Matrix<double> localOnes;

    // compute the rowMeans
    // explicitly compute matrix vector products to avoid relayouts from distributed Gemv!
    El::Ones(localOnes, localRows.Height(), 1);
    El::Gemv(El::TRANSPOSE, 1.0, localRows, localOnes, 0.0, localRowMeans);
    El::Zeros(rowMeans, n, 1);
    rowMeans.Reserve(n);
    for(El::Int col=0; col < n; col++)
        rowMeans.QueueUpdate(col, 1, localRowMeans.Get(col, 1));
    rowMeans.ProcessQueues();

    // subtract off the row means
    El::Ones(distOnes, m, 1);
    El::Ger(-1.0, *matA, distOnes, rowMeans); 

    // compute the column variances
    auto colVariances = El::DistMatrix<double, El::STAR, El::STAR>(n, 1, self->grid);
    auto localSquaredEntries = El::Matrix<double>(localRows.Height(), n);
    auto localColVariances = El::Matrix<double>(n, 1);

    El::Hadamard(localRows, localRows, localSquaredEntries);
    El::Gemv(El::TRANSPOSE, 1.0, localSquaredEntries, localOnes, 0.0, localColVariances);
    El::Zeros(colVariances, n, 1);
    colVariances.Reserve(n);
    for(El::Int col=0; col < n; col++)
        colVariances.QueueUpdate(col, 1, localColVariances.Get(col, 1));
    colVariances.ProcessQueues();

    // rescale by the inv col stdevs
    auto invColStdevs = El::DistMatrix<double, El::STAR, El::STAR>(n ,1, self->grid);
    El::Zeros(invColStdevs, n, 1);
    if(invColStdevs.DistRank() == 0) {
        invColStdevs.Reserve(n);
        for(El::Int col = 0; col < n; ++col) {
            auto curInvStdev = colVariances.Get(col, 1);
            if (curInvStdev < 1e-5) {
                curInvStdev = 1.0;
            } else {
                curInvStdev = 1/sqrt(curInvStdev);
            }
            invColStdevs.QueueUpdate(col, 1, curInvStdev);
        }
    }
    else {
        invColStdevs.Reserve(0);
    }
    invColStdevs.ProcessQueues();

    El::DiagonalScale(El::RIGHT, El::NORMAL, invColStdevs, *matA);
}

}
