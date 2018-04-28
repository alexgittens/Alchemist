#include <iostream>
#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>
#include "utils.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

namespace alchemist{

// only returns the cluster centers, in fitCenters
void kmeansPP(uint32_t seed, std::vector<MatrixXd> points, std::vector<double> weights, MatrixXd & fitCenters, uint32_t maxIters = 30) {
  std::default_random_engine randGen(seed);
  std::uniform_real_distribution<double> unifReal(0.0, 1.0);
  uint32_t n = points.size();
  uint32_t d = points[0].cols();
  uint32_t k = fitCenters.rows();
  std::vector<uint32_t> pointIndices(n);
  std::vector<uint32_t> centerIndices(k);
  std::iota(pointIndices.begin(), pointIndices.end(), 0);
  std::iota(centerIndices.begin(), centerIndices.end(), 0);

  // pick initial cluster center using weighted sampling
  double stopSum = unifReal(randGen)*std::accumulate(weights.begin(), weights.end(), 0.0);
  double curSum = 0.0;
  uint32_t searchIdx = 0;
  while(searchIdx < n && curSum < stopSum) {
    curSum += weights[searchIdx];
    searchIdx += 1;
  }
  fitCenters.row(0) = points[searchIdx - 1];

  // iteratively select next cluster centers with 
  // probability proportional to the squared distance from the previous centers
  // recall we are doing weighted k-means so min sum(w_i*d(x_i,C)) rather than sum(d(x_i,C))


  auto start = std::chrono::system_clock::now();
  std::vector<double> samplingDensity(n);
  for(auto pointIdx : pointIndices) {
    samplingDensity[pointIdx] = weights[pointIdx]*(points[pointIdx] - fitCenters.row(0)).squaredNorm();
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> elapsed_ms(end - start);

  for(uint32_t centerSelectionIdx = 1; centerSelectionIdx < k; centerSelectionIdx++) {
    stopSum = unifReal(randGen)*std::accumulate(samplingDensity.begin(), samplingDensity.end(), 0.0);
    curSum = 0.0; 
    searchIdx = 0;
    while(searchIdx < n && curSum < stopSum) {
      curSum += samplingDensity[searchIdx];
      searchIdx += 1;
    }
    // if less than k initial points explain all the data, set remaining centers to the first point
    fitCenters.row(centerSelectionIdx) = searchIdx > 0 ? points[searchIdx - 1] : points[0]; 
    for(auto pointIdx : pointIndices)
      samplingDensity[pointIdx] = std::min(samplingDensity[pointIdx], 
          weights[pointIdx]*(points[pointIdx] - fitCenters.row(centerSelectionIdx)).squaredNorm());

//    std::cerr << Eigen::Map<Eigen::RowVectorXd>(samplingDensity.data(), n) << std::endl;
  }

  // run Lloyd's algorithm stop when reached max iterations or points stop changing cluster assignments
  bool movedQ;
  std::vector<double> clusterSizes(k, 0.0);
  std::vector<uint32_t> clusterAssignments(n, 0);
  MatrixXd clusterPointSums(k, d);
  VectorXd distanceSqToCenters(k);
  uint32_t newClusterAssignment;
  double sqDist;

  uint32_t iter = 0;
  for(; iter < maxIters; iter++) {
    movedQ = false;
    clusterPointSums.setZero();
    std::fill(clusterSizes.begin(), clusterSizes.end(), 0);

    // assign each point to nearest cluster and count number of points in each cluster
    for(auto pointIdx : pointIndices) {
      for(auto centerIdx : centerIndices)
        distanceSqToCenters(centerIdx) = (points[pointIdx] - fitCenters.row(centerIdx)).squaredNorm();
      sqDist = distanceSqToCenters.minCoeff(&newClusterAssignment);
      if (newClusterAssignment != clusterAssignments[pointIdx])
        movedQ = true;
      clusterAssignments[pointIdx] = newClusterAssignment;
      clusterPointSums.row(newClusterAssignment) += weights[pointIdx]*points[pointIdx];
      clusterSizes[newClusterAssignment] += weights[pointIdx];
    }

    // stop iterations if cluster assignments have not changed
    if(!movedQ) 
      break;

    // update cluster centers
    for(auto centerIdx : centerIndices) {
      if ( clusterSizes[centerIdx] > 0 ) {
        fitCenters.row(centerIdx) = clusterPointSums.row(centerIdx) / clusterSizes[centerIdx];
      } else {
        uint32_t randPtIdx = (uint32_t) std::round(unifReal(randGen)*n);
        fitCenters.row(centerIdx) = points[randPtIdx];
      }
    }
  }

  // seems necessary to force eigen to return the centers as an actual usable matrix
  for(uint32_t rowidx = 0; rowidx < k; rowidx++)
    for(uint32_t colidx = 0; colidx < k; colidx++)
      fitCenters(rowidx, colidx) = fitCenters(rowidx, colidx) + 0.0;
}

uint32_t updateAssignmentsAndCounts(MatrixXd const & dataMat, MatrixXd const & centers,
    uint32_t * clusterSizes, std::vector<uint32_t> & rowAssignments, double & objVal) {
  uint32_t numCenters = centers.rows();
  VectorXd distanceSq(numCenters);
  El::Int newAssignment;
  uint32_t numChanged = 0;
  objVal = 0.0;

  for(uint32_t idx = 0; idx < numCenters; ++idx)
    clusterSizes[idx] = 0;

  for(El::Int rowIdx = 0; rowIdx < dataMat.rows(); ++rowIdx) {
    for(uint32_t centerIdx = 0; centerIdx < numCenters; ++centerIdx)
      distanceSq[centerIdx] = (dataMat.row(rowIdx) - centers.row(centerIdx)).squaredNorm();
    objVal += distanceSq.minCoeff(&newAssignment);
    if (rowAssignments[rowIdx] != newAssignment)
      numChanged += 1;
    rowAssignments[rowIdx] = newAssignment;
    clusterSizes[rowAssignments[rowIdx]] += 1;
  }

  return numChanged;
}

// TODO: add seed as argument (make sure different workers do different things)
void kmeansParallelInit(Worker * self, DistMatrix const * dataMat,
    MatrixXd const & localData, uint32_t scale, uint32_t initSteps, MatrixXd & clusterCenters, uint64_t seed) {

  auto d = localData.cols();

  // if you have the initial cluster seed, send it to everyone
  uint32_t rowIdx;
  mpi::broadcast(self->world, rowIdx, 0);
  MatrixXd initialCenter;

  if (dataMat->IsLocalRow(rowIdx)) {
    auto localRowIdx = dataMat->LocalRow(rowIdx);
    initialCenter = localData.row(localRowIdx);
    int maybe_root = 1;
    int rootProcess;
    mpi::all_reduce(self->peers, self->peers.rank(), rootProcess, std::plus<int>());
    mpi::broadcast(self->peers, initialCenter, self->peers.rank());
  }
  else {
    int maybe_root = 0;
    int rootProcess;
    mpi::all_reduce(self->peers, 0, rootProcess, std::plus<int>());
    mpi::broadcast(self->peers, initialCenter, rootProcess);
  }

  //in each step, sample 2*k points on average (totalled across the partitions)
  // with probability proportional to their squared distance from the current
  // cluster centers and add the sampled points to the set of cluster centers
  std::vector<double> distSqToCenters(localData.rows());
  double Z; // normalization constant
  std::mt19937 gen(seed + self->world.rank());
  std::uniform_real_distribution<double> dis(0, 1);

  std::vector<MatrixXd> initCenters;
  initCenters.push_back(initialCenter);

  for(int steps = 0; steps < initSteps; ++steps) {
    // 1) compute the distance of your points from the set of centers and all_reduce
    // to get the normalization for the sampling probability
    VectorXd distSq(initCenters.size());
    Z = 0;
    for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
      for(uint32_t centerIdx = 0; centerIdx < initCenters.size(); ++centerIdx)
        distSq[centerIdx] = (localData.row(pointIdx) - initCenters[centerIdx]).squaredNorm();
      distSqToCenters[pointIdx] = distSq.minCoeff();
      Z += distSqToCenters[pointIdx];
    }
    mpi::all_reduce(self->peers, mpi::inplace_t<double>(Z), std::plus<double>());

    // 2) sample your points accordingly
    std::vector<MatrixXd> localNewCenters;
    for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
      bool sampledQ = ( dis(gen) < ((double)scale) * distSqToCenters[pointIdx]/Z ) ? true : false;
      if (sampledQ) {
        localNewCenters.push_back(localData.row(pointIdx));
      }
    };

    // 3) have each worker broadcast out their sampled points to all other workers,
    // to update each worker's set of centers
    for(uint32_t root= 0; root < self->peers.size(); ++root) {
      if (root == self->peers.rank()) {
        mpi::broadcast(self->peers, localNewCenters, root);
        initCenters.insert(initCenters.end(), localNewCenters.begin(), localNewCenters.end());
      } else {
        std::vector<MatrixXd> remoteNewCenters;
        mpi::broadcast(self->peers, remoteNewCenters, root);
        initCenters.insert(initCenters.end(), remoteNewCenters.begin(), remoteNewCenters.end());
      }
    }
  } // end for

  // figure out the number of points closest to each cluster center
  std::vector<uint32_t> clusterSizes(initCenters.size(), 0);
  std::vector<uint32_t> localClusterSizes(initCenters.size(), 0);
  VectorXd distSq(initCenters.size());
  for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
    for(uint32_t centerIdx = 0; centerIdx < initCenters.size(); ++centerIdx)
      distSq[centerIdx] = (localData.row(pointIdx) - initCenters[centerIdx]).squaredNorm();
    uint32_t clusterIdx;
    distSq.minCoeff(&clusterIdx);
    localClusterSizes[clusterIdx] += 1;
  }

  mpi::all_reduce(self->peers, localClusterSizes.data(), localClusterSizes.size(),
      clusterSizes.data(), std::plus<uint32_t>());

  // after centers have been sampled, sync back up with the driver,
  // and send them there for local clustering
  self->world.barrier();
  if(self->world.rank() == 1) {
    self->world.send(0, 0, clusterSizes);
    self->world.send(0, 0, initCenters);
  }
  self->world.barrier();

  clusterCenters.setZero();
  mpi::broadcast(self->world, clusterCenters.data(), clusterCenters.rows()*d, 0);
}


// TODO: add seed as argument (make sure different workers do different things)
void KMeansCommand::run(Worker *self) const {
  auto log = self->log;
  log->info("Started kmeans");
  auto origDataMat = self->matrices[origMat].get();
  auto n = origDataMat->Height();
  auto d = origDataMat->Width();

  // relayout matrix if needed so that it is in row-partitioned format
  // cf http://libelemental.org/pub/slides/ICS13.pdf slide 19 for the cost of redistribution
  auto distData = origDataMat->DistData();
  DistMatrix * dataMat = new El::DistMatrix<double, El::VR, El::STAR>(n, d, self->grid);
  if (distData.colDist == El::VR && distData.rowDist == El::STAR) {
   dataMat = origDataMat;
  } else {
    auto relayoutStart = std::chrono::system_clock::now();
    El::Copy(*origDataMat, *dataMat); // relayouts data so it is row-wise partitioned
    std::chrono::duration<double, std::milli> relayoutDuration(std::chrono::system_clock::now() - relayoutStart);
    log->info("Detected matrix is not row-partitioned, so relayouted to row-partitioned; took {} ms ", relayoutDuration.count());
  }

  // TODO: store these as local matrices on the driver
  DistMatrix * centers = new El::DistMatrix<double, El::VR, El::STAR>(numCenters, d, self->grid);
  DistMatrix * assignments = new El::DistMatrix<double, El::VR, El::STAR>(n, 1, self->grid);
  ENSURE(self->matrices.insert(std::make_pair(centersHandle, std::unique_ptr<DistMatrix>(centers))).second);
  ENSURE(self->matrices.insert(std::make_pair(assignmentsHandle, std::unique_ptr<DistMatrix>(assignments))).second);

  MatrixXd localData(dataMat->LocalHeight(), d);

  // compute the map from local row indices to the row indices in the global matrix
  // and populate the local data matrix

  std::vector<El::Int> rowMap(localData.rows());
  for(El::Int rowIdx = 0; rowIdx < n; ++rowIdx)
    if (dataMat->IsLocalRow(rowIdx)) {
      auto localRowIdx = dataMat->LocalRow(rowIdx);
      rowMap[localRowIdx] = rowIdx;
      for(El::Int colIdx = 0; colIdx < d; ++colIdx)
        localData(localRowIdx, colIdx) = dataMat->GetLocal(localRowIdx, colIdx);
    }

  MatrixXd clusterCenters(numCenters, d);
  MatrixXd oldClusterCenters(numCenters, d);

  // initialize centers using kMeans||
  uint32_t scale = 2*numCenters;
  clusterCenters.setZero();
  kmeansParallelInit(self, dataMat, localData, scale, initSteps, clusterCenters, seed);

  // TODO: allow to initialize k-means randomly
  //MatrixXd clusterCenters = MatrixXd::Random(numCenters, d);

  /******** START Lloyd's iterations ********/
  // compute the local cluster assignments
  std::unique_ptr<uint32_t[]> counts{new uint32_t[numCenters]};
  std::vector<uint32_t> rowAssignments(localData.rows());
  VectorXd distanceSq(numCenters);
  double objVal;

  updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);

  MatrixXd centersBuf(numCenters, d);
  std::unique_ptr<uint32_t[]> countsBuf{new uint32_t[numCenters]};
  uint32_t numChanged = 0;
  oldClusterCenters = clusterCenters;

  while(true) {
    uint32_t nextCommand;
    mpi::broadcast(self->world, nextCommand, 0);

    if (nextCommand == 0xf)  // finished iterating
      break;
    else if (nextCommand == 2) { // encountered an empty cluster, so randomly pick a point in the dataset as that cluster's centroid
      uint32_t clusterIdx, rowIdx;
      mpi::broadcast(self->world, clusterIdx, 0);
      mpi::broadcast(self->world, rowIdx, 0);
      if (dataMat->IsLocalRow(rowIdx)) {
        auto localRowIdx = dataMat->LocalRow(rowIdx);
        clusterCenters.row(clusterIdx) = localData.row(localRowIdx);
      }
      mpi::broadcast(self->peers, clusterCenters, self->peers.rank());
      updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);
      self->world.barrier();
      continue;
    }

    /******** do a regular Lloyd's iteration ********/

    // update the centers
    // TODO: locally compute cluster sums and place in clusterCenters
    oldClusterCenters = clusterCenters;
    clusterCenters.setZero();
    for(uint32_t rowIdx = 0; rowIdx < localData.rows(); ++rowIdx)
      clusterCenters.row(rowAssignments[rowIdx]) += localData.row(rowIdx);

    mpi::all_reduce(self->peers, clusterCenters.data(), numCenters*d, centersBuf.data(), std::plus<double>());
    std::memcpy(clusterCenters.data(), centersBuf.data(), numCenters*d*sizeof(double));
    mpi::all_reduce(self->peers, counts.get(), numCenters, countsBuf.get(), std::plus<uint32_t>());
    std::memcpy(counts.get(), countsBuf.get(), numCenters*sizeof(uint32_t));

    for(uint32_t rowIdx = 0; rowIdx < numCenters; ++rowIdx)
      if( counts[rowIdx] > 0)
        clusterCenters.row(rowIdx) /= counts[rowIdx];

    // compute new local assignments
    numChanged = updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);
    std::cerr << "computed Updated assingments\n" << std::flush;

    // return the number of changed assignments
    mpi::reduce(self->world, numChanged, std::plus<int>(), 0);
    // return the cluster counts
    mpi::reduce(self->world, counts.get(), numCenters, std::plus<uint32_t>(), 0);
    std::cerr << "returned cluster counts\n" << std::flush;
    if (self->world.rank() == 1) {
      bool movedQ = (clusterCenters - oldClusterCenters).rowwise().norm().minCoeff() > changeThreshold;
      self->world.send(0, 0, movedQ);
    }
  }

  // write the final k-means centers and assignments
  auto startKMeansWrite = std::chrono::system_clock::now();
  El::Zero(*assignments);
  assignments->Reserve(localData.rows());
  for(El::Int rowIdx = 0; rowIdx < localData.rows(); ++rowIdx)
    assignments->QueueUpdate(rowMap[rowIdx], 0, rowAssignments[rowIdx]);
  assignments->ProcessQueues();

  El::Zero(*centers);
  centers->Reserve(centers->LocalHeight()*d);
  for(uint32_t clusterIdx = 0; clusterIdx < numCenters; ++clusterIdx)
    if (centers->IsLocalRow(clusterIdx)) {
      for(El::Int colIdx = 0; colIdx < d; ++colIdx)
        centers->QueueUpdate(clusterIdx, colIdx, clusterCenters(clusterIdx, colIdx));
    }
  centers->ProcessQueues();
  std::chrono::duration<double, std::milli> kMeansWrite_duration(std::chrono::system_clock::now() - startKMeansWrite);
  std::cerr << self->world.rank() << ": writing the k-means centers and assignments took " << kMeansWrite_duration.count() << "ms\n";

  mpi::reduce(self->world, objVal, std::plus<double>(), 0);
  self->world.barrier();
}

} // end namespace alchemist
