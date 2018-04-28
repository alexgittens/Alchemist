#include "alchemist.h"

namespace alchemist {

using Eigen::MatrixXd;
using Eigen::VectorXd;

void TruncatedSVDCommand::run(Worker *self) const {
  auto m = self->matrices[mat]->Height();
  auto n = self->matrices[mat]->Width();
  auto workingMat = self->matrices[mat].get();

  int LOCALEIGS = 0; // TODO: make these an enumeration, and global to Alchemist
  int LOCALEIGSPRECOMPUTE = 1;
  int DISTEIGS = 2; 

  // Assume matrix is row-partitioned b/c relaying it out doubles memory requirements

  //NB: sometimes it makes sense to precompute the gramMat (when it's cheap (we have a lot of cores and enough memory), sometimes
  // it makes more sense to compute A'*(A*x) separately each time (when we don't have enough memory for gramMat, or its too expensive
  // time-wise to precompute GramMat). trade-off depends on k (through the number of Arnoldi iterations we'll end up needing), the
  // amount of memory we have free to store GramMat, and the number of cores we have available
  El::Matrix<double> localGramChunk;

  if (method == LOCALEIGSPRECOMPUTE) {
      localGramChunk.Resize(n, n);
      self->log->info("Computing the local contribution to A'*A");
      self->log->info("Local matrix's dimensions are {}, {}", workingMat->LockedMatrix().Height(), workingMat->LockedMatrix().Width());
      self->log->info("Storing A'*A in {},{} matrix", n, n);
      auto startFillLocalMat = std::chrono::system_clock::now();
      if (workingMat->LockedMatrix().Height() > 0)
        El::Gemm(El::TRANSPOSE, El::NORMAL, 1.0, workingMat->LockedMatrix(), workingMat->LockedMatrix(), 0.0, localGramChunk);
      else
        El::Zeros(localGramChunk, n, n);
      std::chrono::duration<double, std::milli> fillLocalMat_duration(std::chrono::system_clock::now() - startFillLocalMat);
      self->log->info("Took {} ms to compute local contribution to A'*A", fillLocalMat_duration.count());
  }

  uint32_t command;
  std::unique_ptr<double[]> vecIn{new double[n]};
  El::Matrix<double> localx(n, 1);
  El::Matrix<double> localintermed(workingMat->LocalHeight(), 1);
  El::Matrix<double> localy(n, 1);
  localx.LockedAttach(n, 1, vecIn.get(), 1);
  auto distx = El::DistMatrix<double, El::STAR, El::STAR>(n, 1, self->grid);
  auto distintermed = El::DistMatrix<double, El::STAR, El::STAR>(m, 1, self->grid);

  self->log->info("finished initialization for truncated SVD");

  while(true) {
    mpi::broadcast(self->world, command, 0);
    if (command == 1 && method == LOCALEIGS) {
        mpi::broadcast(self->world, vecIn.get(), n, 0);
        El::Gemv(El::NORMAL, 1.0, workingMat->LockedMatrix(), localx, 0.0, localintermed);
        El::Gemv(El::TRANSPOSE, 1.0, workingMat->LockedMatrix(), localintermed, 0.0, localy);
        mpi::reduce(self->world, localy.LockedBuffer(), n, std::plus<double>(), 0);
    }
    if (command == 1 && method == LOCALEIGSPRECOMPUTE) {
      mpi::broadcast(self->world, vecIn.get(), n, 0);
      El::Gemv(El::NORMAL, 1.0, localGramChunk, localx, 0.0, localy);
      mpi::reduce(self->world, localy.LockedBuffer(), n, std::plus<double>(), 0);
    } 
    if (command == 1 && method == DISTEIGS) {
      El::Zeros(distx, n, 1);
      self->log->info("Computing a mat-vec prod against A^TA");
      if(self->world.rank() == 1) {
          self->world.recv(0, 0, vecIn.get(), n);
          distx.Reserve(n);
          for(El::Int row=0; row < n; row++)
              distx.QueueUpdate(row, 0, vecIn[row]);
      }
      else {
          distx.Reserve(0);
      }
      distx.ProcessQueues();
      self->log->info("Retrieved x, computing A^TAx");
      El::Gemv(El::NORMAL, 1.0, *workingMat, distx, 0.0, distintermed);
      self->log->info("Computed y = A*x");
      El::Gemv(El::TRANSPOSE, 1.0, *workingMat, distintermed, 0.0, distx);
      self->log->info("Computed x = A^T*y");
      if(self->world.rank() == 1) {
          self->world.send(0, 0, distx.LockedBuffer(), n);
      }
    }
    if (command == 2) {
      uint32_t nconv;
      mpi::broadcast(self->world, nconv, 0);

      MatrixXd rightEigs(n, nconv);
      mpi::broadcast(self->world, rightEigs.data(), n*nconv, 0);
      VectorXd singValsSq(nconv);
      mpi::broadcast(self->world, singValsSq.data(), nconv, 0);
      self->log->info("Received the right eigenvectors and the eigenvalues");

      auto U = new El::DistMatrix<double, El::VR, El::STAR>(m, nconv, self->grid);
      DistMatrix * S = new El::DistMatrix<double, El::VR, El::STAR>(nconv, 1, self->grid);
      DistMatrix * Sinv = new El::DistMatrix<double, El::VR, El::STAR>(nconv, 1, self->grid);
      DistMatrix * V = new El::DistMatrix<double, El::VR, El::STAR>(n, nconv, self->grid);

      ENSURE(self->matrices.insert(std::make_pair(UHandle, std::unique_ptr<DistMatrix>(U))).second);
      ENSURE(self->matrices.insert(std::make_pair(SHandle, std::unique_ptr<DistMatrix>(S))).second);
      ENSURE(self->matrices.insert(std::make_pair(VHandle, std::unique_ptr<DistMatrix>(V))).second);
      self->log->info("Created new matrix objects to hold U,S,V");

      // populate V
      for(El::Int rowIdx=0; rowIdx < n; rowIdx++)
        for(El::Int colIdx=0; colIdx < (El::Int) nconv; colIdx++)
          if(V->IsLocal(rowIdx, colIdx))
            V->SetLocal(V->LocalRow(rowIdx), V->LocalCol(colIdx), rightEigs(rowIdx,colIdx));
      rightEigs.resize(0,0); // clear any memory this temporary variable used (a lot, since it's on every rank)

      // populate S, Sinv
      for(El::Int idx=0; idx < (El::Int) nconv; idx++) {
        if(S->IsLocal(idx, 0))
          S->SetLocal(S->LocalRow(idx), 0, std::sqrt(singValsSq(idx)));
        if(Sinv->IsLocal(idx, 0))
          Sinv->SetLocal(Sinv->LocalRow(idx), 0, 1/std::sqrt(singValsSq(idx)));
      }
      self->log->info("Stored V and S");

      // form U
      self->log->info("computing A*V = U*Sigma");
      self->log->info("A is {}-by-{}, V is {}-by-{}, the resulting matrix should be {}-by-{}", workingMat->Height(), workingMat->Width(), V->Height(), V->Width(), U->Height(), U->Width());
      //Gemm(1.0, *workingMat, *V, 0.0, *U, self->log);
      El::Gemm(El::NORMAL, El::NORMAL, 1.0, *workingMat, *V, 0.0, *U);
      self->log->info("done computing A*V, rescaling to get U");
      // TODO: do a QR instead to ensure stability, but does column pivoting so would require postprocessing S,V to stay consistent
      El::DiagonalScale(El::RIGHT, El::NORMAL, *Sinv, *U);
      self->log->info("Computed and stored U");

      break;
    }
  }

  self->world.barrier();
}

}
