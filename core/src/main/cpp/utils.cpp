#include "alchemist.h"
#include "utils.h"

namespace alchemist {

El::DistMatrix<double> * relayout(const El::DistMatrix<double, El::VR, El::STAR> & matIn, const El::Grid & grid) {
  El::DistMatrix<double> * matOut = new El::DistMatrix<double>(matIn.Height(), matIn.Width(), grid);
  El::Copy(matIn, *matOut);
  return matOut;
}

El::DistMatrix<double, El::VR, El::STAR> * delayout(const El::DistMatrix<double> & matIn, const El::Grid & grid) {
  auto matOut = new El::DistMatrix<double, El::VR, El::STAR>(matIn.Height(), matIn.Width(), grid);
  El::Copy(matIn, *matOut);
  return matOut;
}

// custom GEMM for multiplying [VR,STAR] matrices in normal orientation and storing as [VR, STAR]
// overhead: additional maxPanelSize GB (1 for now) per rank => at most numcores * maxPanelGB per machine
void Gemm(double alpha, const El::DistMatrix<double, El::VR, El::STAR> & A, const El::DistMatrix<double, El::VR, El::STAR> & B,
    double beta, El::DistMatrix<double, El::VR, El::STAR> & C, std::shared_ptr<spdlog::logger> & log) {
  auto m = A.Height();
  auto n = A.Width();
  auto k = B.Width();
  assert(n == B.Height());

  El::Int maxPanelSize = 1; // maximum panel size (will be stored on each process) in GB
  El::Int maxPanelWidth = std::max( (int) std::floor( (maxPanelSize*1000*1000*1000)/((double)8*n) ), 1);
  El::Int numPanels = (int) std::ceil(k/(double)maxPanelWidth);
  log->info("Using {} panels", numPanels);

  El::DistMatrix<double, El::STAR, El::STAR> curBPanel(B.Grid());
  El::Matrix<double> curCPanel;
  for(int curPanelNum = 0; curPanelNum < numPanels; ++curPanelNum) {
    El::Int startCol = curPanelNum*maxPanelWidth;
    El::Int lastCol = std::min(startCol + maxPanelWidth - 1, k - 1);
    El::Int curPanelWidth = lastCol - startCol + 1;

    log->info("Creating next B panel");
    El::Zeros(curBPanel, n, curPanelWidth);
    curBPanel.Reserve(curBPanel.LocalHeight()*curPanelWidth);
    for(El::Int row = 0; row < B.LocalHeight(); ++row)
      for(El::Int col = startCol; col <= lastCol; ++col)
        curBPanel.QueueUpdate(B.GlobalRow(row), col, B.LockedMatrix().Get(row, col));
    curBPanel.ProcessQueues();
    log->info("Finished creating current B panel");

    log->info("Creating next C panel");
    El::View(curCPanel, C.Matrix(), El::Range<El::Int>(0, C.LocalHeight()), El::Range<El::Int>(startCol, lastCol+1));
    log->info("Finished creating current C panel");
    log->info("Multiplying A by current B panel, storing into current C panel");
    El::Gemm(El::NORMAL, El::NORMAL, alpha, A.LockedMatrix(), curBPanel.LockedMatrix(), beta, curCPanel);
    log->info("Done storing current C panel");
  }
}

}
