#include "alchemist.h"

namespace alchemist {

void Gemm(double alpha, const El::DistMatrix<double, El::VR, El::STAR> & A, const El::DistMatrix<double, El::VR, El::STAR> & B,
    double beta, El::DistMatrix<double, El::VR, El::STAR> & C, std::shared_ptr<spdlog::logger> & log); 

El::DistMatrix<double> * relayout(const El::DistMatrix<double, El::VR, El::STAR> & matIn, const El::Grid & grid); 

El::DistMatrix<double, El::VR, El::STAR> * delayout(const El::DistMatrix<double> & matIn, const El::Grid & grid);

}
