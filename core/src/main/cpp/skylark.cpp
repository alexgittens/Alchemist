#include "alchemist.h"
#include <skylark.hpp>
#include "factorizedCG.hpp"
#include "hilbert.hpp"
#include "utils.h"

namespace alchemist {

// NB: it seems that Skylark's LSQR routine cannot work with VR, STAR matrices (get templating errors from their GEMM routine),
// so need to relayout the input and output matrices
void SkylarkLSQRSolverCommand::run(Worker *self) const {
  auto log = self->log;
  El::DistMatrix<double, El::VR, El::STAR> * Amat = (El::DistMatrix<double, El::VR, El::STAR> *) self->matrices[A].get();
  El::DistMatrix<double, El::VR, El::STAR> * Bmat = (El::DistMatrix<double, El::VR, El::STAR> *) self->matrices[B].get();
  El::DistMatrix<double> * Xmat = new El::DistMatrix<double>(Amat->Width(), Bmat->Width(), self->grid);

  log->info("Relaying out lhs and rhs for LSQR");
  auto startRelayout = std::chrono::system_clock::now();
  El::DistMatrix<double> * Arelayedout = relayout(*Amat, self->grid);
  El::DistMatrix<double> * Brelayedout = relayout(*Bmat, self->grid);
  std::chrono::duration<double, std::milli> relayoutDuration(std::chrono::system_clock::now() - startRelayout);
  log->info("Relayout took {} seconds", relayoutDuration.count()/1000.0);

  auto params = skylark::algorithms::krylov_iter_params_t(tolerance, iter_lim);
  skylark::algorithms::LSQR(*Arelayedout, *Brelayedout, *Xmat, params);
  Arelayedout->EmptyData();
  Brelayedout->EmptyData();
  El::DistMatrix<double, El::VR, El::STAR> * Xrelayedout = delayout(*Xmat, self->grid);
  Xmat->EmptyData();

  log->info("LSQR result has dimension {}-by-{}", Xrelayedout->Height(), Xrelayedout->Width());
  ENSURE(self->matrices.insert(std::make_pair(X, std::unique_ptr<DistMatrix>(Xrelayedout))).second);
  self->world.barrier();
}

// Note if we do regression, we can only have one rhs
// The call to the ADMM solver follows  the template of the LargeScaleKernelLearning function in hilbert.hpp
void SkylarkKernelSolverCommand::run(Worker *self) const {

  self->log->info("Setting up solver options");

  // set some options that aren't currently passed in as arguments
  bool usefast = false;
  SequenceType seqtype = LEAPED_HALTON;
  int numthreads = 1;
  bool cachetransforms = true;

  skylark::base::context_t context(seed);
  typedef El::Matrix<double> InputType;

  El::Matrix<double> localX;
  El::Matrix<double> localY;

  El::Transpose(self->matrices[features].get()->LockedMatrix(), localX);
  El::Transpose(self->matrices[targets].get()->LockedMatrix(), localY);

  // validation data
  El::Matrix<double> localXv;
  El::Matrix<double> localYv;

  // don't know what the variable targets is used for
  // shift indicates whether validation data should also be shifted
  auto dimensions = skylark::base::Height(localX);
  auto targets = regression ? 1 : GetNumTargets(self->peers, localY);
  bool shift = false;

  if (!regression && lossfunction == LOGISTIC && targets == 1) {
    ShiftForLogistic(localY);
    targets = 2;
    shift = true;
  }

  self->log->info("Setting up Skylark ADMM solver");

  skylark::algorithms::loss_t<double> *loss = NULL;
  switch(lossfunction) {
    case SQUARED:
        loss = new skylark::algorithms::squared_loss_t<double>();
        break;
    case HINGE:
        loss = new skylark::algorithms::hinge_loss_t<double>();
        break;
    case LOGISTIC:
        loss = new skylark::algorithms::logistic_loss_t<double>();
        break;
    case LAD:
        loss = new skylark::algorithms::lad_loss_t<double>();
        break;
		default:
				break;
  }

	skylark::algorithms::regularizer_t<double> *reg = NULL;
	if (lambda == 0 || regularizer == NOREG)
			reg = new skylark::algorithms::empty_regularizer_t<double>();
	else
			switch(regularizer) {
				case L2:
						reg = new skylark::algorithms::l2_regularizer_t<double>();
						break;
				case L1:
						reg = new skylark::algorithms::l1_regularizer_t<double>();
						break;
				default:
						break;
			}

  self->log->info("Initializing solver");

  BlockADMMSolver<InputType> *Solver = NULL;
    int features = 0;
    switch(kernel) {
    case K_LINEAR:
        features =
            (randomfeatures == 0 ? dimensions : randomfeatures);
        if (randomfeatures == 0) {
            Solver =
                new BlockADMMSolver<InputType>(loss,
                    reg,
                    lambda,
                    dimensions,
                    numfeaturepartitions);
          }
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    reg,
                    lambda,
                    features,
                    skylark::ml::linear_t(dimensions),
                    skylark::ml::sparse_feature_transform_tag(),
                    numfeaturepartitions);
        break;

    case K_GAUSSIAN:
        features = randomfeatures;
        if (!usefast)
            if (seqtype == LEAPED_HALTON)
                Solver =
                    new BlockADMMSolver<InputType>(context,
                        loss,
                        reg,
                        lambda,
                        features,
                        skylark::ml::gaussian_t(dimensions,
                            kernelparam),
                        skylark::ml::quasi_feature_transform_tag(),
                        numfeaturepartitions);
            else
                Solver =
                    new BlockADMMSolver<InputType>(context,
                        loss,
                        reg,
                        lambda,
                        features,
                        skylark::ml::gaussian_t(dimensions,
                            kernelparam),
                        skylark::ml::regular_feature_transform_tag(),
                        numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    reg,
                    lambda,
                    features,
                    skylark::ml::gaussian_t(dimensions,
                        kernelparam),
                    skylark::ml::fast_feature_transform_tag(),
                    numfeaturepartitions);
        break;

    case K_POLYNOMIAL:
        features = randomfeatures;
        Solver =
            new BlockADMMSolver<InputType>(context,
                loss,
                reg,
                lambda,
                features,
                skylark::ml::polynomial_t(dimensions,
                    kernelparam, kernelparam2, kernelparam3),
                skylark::ml::regular_feature_transform_tag(),
                numfeaturepartitions);
        break;

    case K_MATERN:
        features = randomfeatures;
        if (!usefast)
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    reg,
                    lambda,
                    features,
                    skylark::ml::matern_t(dimensions,
                        kernelparam, kernelparam2),
                    skylark::ml::regular_feature_transform_tag(),
                    numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    reg,
                    lambda,
                    features,
                    skylark::ml::matern_t(dimensions,
                        kernelparam, kernelparam2),
                    skylark::ml::fast_feature_transform_tag(),
                    numfeaturepartitions);
        break;

    case K_LAPLACIAN:
        features = randomfeatures;
        if (seqtype == LEAPED_HALTON)
            new BlockADMMSolver<InputType>(context,
                loss,
                reg,
                lambda,
                features,
                skylark::ml::laplacian_t(dimensions,
                    kernelparam),
                skylark::ml::quasi_feature_transform_tag(),
                numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    reg,
                    lambda,
                    features,
                    skylark::ml::laplacian_t(dimensions,
                        kernelparam),
                    skylark::ml::regular_feature_transform_tag(),
                    numfeaturepartitions);

        break;

    case K_EXPSEMIGROUP:
        features = randomfeatures;
        if (seqtype == LEAPED_HALTON)
            new BlockADMMSolver<InputType>(context,
                loss,
                reg,
                lambda,
                features,
                skylark::ml::expsemigroup_t(dimensions,
                    kernelparam),
                skylark::ml::quasi_feature_transform_tag(),
                numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    reg,
                    lambda,
                    features,
                    skylark::ml::expsemigroup_t(dimensions,
                        kernelparam),
                    skylark::ml::regular_feature_transform_tag(),
                    numfeaturepartitions);
        break;

    default:
        // TODO!
        break;

    }

    self->log->info("Solver initialized");

    // Set parameters
    Solver->set_rho(rho);
    Solver->set_maxiter(maxiter);
    Solver->set_tol(tolerance);
    Solver->set_nthreads(numthreads);
    Solver->set_cache_transform(cachetransforms);

  self->log->info("Training solver");
  skylark::ml::hilbert_model_t * model = Solver->train(localX, localY, localXv, localYv, regression, self->peers);
  self->log->info("Finished training, now retreiving coefficients");
  El::Matrix<double> X = model->get_coef();

  // Convert the model coefficients to a distributed matrix and store in the matrix table
  DistMatrix * Xdist = new El::DistMatrix<double, El::VR, El::STAR>(X.Height(), X.Width(), self->grid);
  for(uint32_t row = 0; row < Xdist->Height(); row++) 
    if (Xdist->IsLocalRow(row)) 
      for (uint32_t col = 0; col < Xdist->Width(); col++)
        Xdist->Set(row, col, X.Get(row,col));

  ENSURE(self->matrices.insert(std::make_pair(coefs, std::unique_ptr<DistMatrix>(Xdist))).second);
  self->log->info("Stored the coefficient matrix as matrix {}", coefs);

  self->world.barrier();
}

void RandomFourierFeaturesCommand::run(Worker *self) const {
    auto log = self->log;
    typedef El::DistMatrix<double, El::VR, El::STAR> DistMatrixType;
    namespace skys = skylark::sketch;

    DistMatrixType * Amat = (DistMatrixType *) self->matrices[A].get();
    DistMatrixType * Fmat = new DistMatrixType(Amat->Height(), numRandFeatures, self->grid);

    skylark::base::context_t context(seed);
    skys::GaussianRFT_t<DistMatrixType, DistMatrixType> RFFSketcher(Amat->Width(), numRandFeatures, sigma, context);

    log->info("Computing the Gaussian Random Features");
    RFFSketcher.apply(*Amat, *Fmat, skys::rowwise_tag());
    log->info("Finished computing");
    ENSURE(self->matrices.insert(std::make_pair(X, std::unique_ptr<DistMatrix>(Fmat))).second);

    self->world.barrier();
}

void FactorizedCGSolverCommand::run(Worker *self) const {
  auto log = self->log;
  El::DistMatrix<double, El::VR, El::STAR> * Amat = (El::DistMatrix<double, El::VR, El::STAR> *) self->matrices[A].get();
  El::DistMatrix<double, El::VR, El::STAR> * Bmat = (El::DistMatrix<double, El::VR, El::STAR> *) self->matrices[B].get();
  El::DistMatrix<double> * Xmat = new El::DistMatrix<double>(Amat->Width(), Bmat->Width(), self->grid);
  El::Bernoulli(*Xmat, Xmat->Height(), Xmat->Width());

  log->info("Relaying out lhs and rhs for CG solver");
  auto startRelayout = std::chrono::system_clock::now();
  El::DistMatrix<double> * Arelayedout = relayout(*Amat, self->grid);
  El::DistMatrix<double> * Brelayedout = relayout(*Bmat, self->grid);
  std::chrono::duration<double, std::milli> relayoutDuration(std::chrono::system_clock::now() - startRelayout);
  log->info("Relayout took {} seconds", relayoutDuration.count()/1000.0);

  auto params = skylark::algorithms::krylov_iter_params_t();
  params.iter_lim = maxIters;
  params.am_i_printing = true;
  params.log_level = 2;
  params.res_print = 20;
  log->info("Calling the CG solver");
  skylark::algorithms::factorizedCG(*Arelayedout, *Brelayedout, *Xmat, lambda, log, params);
  Arelayedout->EmptyData();
  Brelayedout->EmptyData();
  El::DistMatrix<double, El::VR, El::STAR> * Xrelayedout = delayout(*Xmat, self->grid);
  Xmat->EmptyData();

  log->info("CG results has dimensions {}-by-{}", Xrelayedout->Height(), Xrelayedout->Width());
  ENSURE(self->matrices.insert(std::make_pair(X, std::unique_ptr<DistMatrix>(Xrelayedout))).second);
  self->world.barrier();
  
  /*
  El::DistMatrix<double, El::VR, El::STAR> * Xmat = new El::DistMatrix<double, El::VR, El::STAR>(Amat->Width(), Bmat->Width(), self->grid);
  El::Bernoulli(*Xmat, Xmat->Height(), Xmat->Width());

  auto params = skylark::algorithms::krylov_iter_params_t();
  params.iter_lim = maxIters;
  params.am_i_printing = true;
  params.log_level = 2;
  params.res_print = 20;
  log->info("Calling the CG solver");
  skylark::algorithms::factorizedCG(*Amat, *Bmat, *Xmat, lambda, log, params);

  log->info("CG results has dimensions {}-by-{}", Xmat->Height(), Xmat->Width());
  ENSURE(self->matrices.insert(std::make_pair(X, std::unique_ptr<DistMatrix>(Xmat))).second);
  self->world.barrier();
  */
}

}
