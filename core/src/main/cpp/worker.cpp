#include "alchemist.h"
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <poll.h>
#include "data_stream.h"
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include "spdlog/spdlog.h"
#include <time.h>

namespace alchemist {

void MatrixGetWorkerRowsCommand::run(Worker * self) const {
  auto search = self->matrices.find(handle);
  if (search != self->matrices.end()) {
    //self->log->info("Found it!");
  } else {
    self->log->info("Matrix {} not found!", handle.id);
  }
  auto matrix = self->matrices[handle].get();
  auto participatingQ = matrix->LocalHeight() > 0; 
  self->log->info("Returning my local row indices for matrix {}", handle.id);

  if (participatingQ)
    self->log->info("I am participating");
  else
    self->log->info("I am not participating");

  mpi::reduce(self->world, participatingQ ? 1 : 0, std::plus<int>(), 0);

  self->log->info("Creating vector of local rows");
  std::vector<uint64_t> rowIndices;
  for(El::Int rowIdx = 0; rowIdx < matrix->Height(); ++rowIdx)
    if (matrix->IsLocalRow(rowIdx)) {
      rowIndices.push_back(rowIdx);
    }

  for(int workerRank = 1; workerRank < self->world.size(); workerRank++) {
    if(workerRank == self->world.rank()) 
      self->world.send(0 ,0, rowIndices);
    self->world.barrier();
  }
}

void  MatrixGetRowsCommand::run(Worker * self) const {
  auto search = self->matrices.find(handle);
  if (search != self->matrices.end()) {
    //self->log->info("Found it!");
  } else {
    self->log->info("Matrix {} not found!", handle.id);
  }
  auto matrix = self->matrices[handle].get();
  uint64_t numCols = matrix->Width();

  self->log->info("Sending over {} rows from matrix {} with {} cols", matrix->LocalHeight(), handle.id, numCols);

  self->sendMatrixRows(handle, matrix);
  self->world.barrier();
}

void NewMatrixCommand::run(Worker *self) const {
  auto handle = info.handle;
  self->log->info("Creating new distributed matrix");
  DistMatrix *matrix = new El::DistMatrix<double, El::VR, El::STAR>(info.numRows, info.numCols, self->grid);
  Zero(*matrix);
  ENSURE(self->matrices.insert(std::make_pair(handle, std::unique_ptr<DistMatrix>(matrix))).second);
  self->log->info("Created new distributed matrix");

  std::vector<uint64_t> rowsOnWorker;
  self->log->info("Creating vector of local rows");
  rowsOnWorker.reserve(info.numRows);
  std::stringstream ss;
  ALCHEMIST_TRACE(ss << "Local rows: ");
  for(El::Int rowIdx = 0; rowIdx < info.numRows; ++rowIdx)
    if (matrix->IsLocalRow(rowIdx)) {
      rowsOnWorker.push_back(rowIdx);
      ALCHEMIST_TRACE(ss << rowIdx << ' ');
    }
  ALCHEMIST_TRACE(self->log->info(ss.str()));

  for(int workerIdx = 1; workerIdx < self->world.size(); workerIdx++) {
    if( self->world.rank() == workerIdx ) {
      self->world.send(0, 0, rowsOnWorker);
    }
    self->world.barrier();
  }

  self->log->info("Starting to recieve my rows");
  self->receiveMatrixBlocks(handle);
  self->log->info("Received all my matrix rows");
  self->world.barrier();
}

void HaltCommand::run(Worker *self) const {
  self->shouldExit = true;
}

struct WorkerClientSendHandler {
  int sock;
  std::shared_ptr<spdlog::logger> log;
  short pollEvents;
  std::vector<char> inbuf;
  std::vector<char> outbuf;
  std::vector<double> rowbuf;
  size_t inpos;
  size_t outpos;
  const El::AbstractDistMatrix<double> * matrix;
  MatrixHandle handle;
  const size_t numCols;

  // only set POLLOUT when have data to send
  // sends 0x3 code (uint32), then matrix handle (uint32), then row index (long = uint64_t)
  WorkerClientSendHandler(int sock, std::shared_ptr<spdlog::logger> log, MatrixHandle handle, const El::AbstractDistMatrix<double> * matrix, uint64_t numCols) :
    sock(sock), log(log), pollEvents(POLLIN), inbuf(16), outbuf(8 + numCols * 8), rowbuf(numCols), inpos(0), outpos(0),
    matrix(matrix), handle(handle), numCols(numCols) {
  }

  ~WorkerClientSendHandler() {
    close();
  }

  // note this is never used! (it should be, to remove the client from the set of clients being polled once the operation on that client is done
  bool isClosed() const {
    return sock == -1;
  }

  void close() {
    if(sock != -1) ::close(sock);
    sock = -1;
    pollEvents = 0;
  }

  int handleEvent(short revents) {
    mpi::communicator world;
    int rowsCompleted = 0;

    // handle reads: Spark asks for a row from a given matrix handle
    if(revents & POLLIN && pollEvents & POLLIN) {
      while(!isClosed()) {
        int count = recv(sock, &inbuf[inpos], inbuf.size() - inpos, 0);
        //std::cerr << format("%s: read: sock=%s, inbuf=%s, inpos=%s, count=%s\n")
        //    % world.rank() % sock % inbuf.size() % inpos % count;
        if (count == 0) {
          // means the other side has closed the socket
          break;
        } else if( count == -1) {
          if(errno == EAGAIN) {
            // no more input available until next POLLIN
            break;
          } else if(errno == EINTR) {
            // interrupted (e.g. by signal), so try again
            continue;
          } else if(errno == ECONNRESET) {
            close();
            break;
          } else {
            // TODO
            abort();
          }
        } else {
          ENSURE(count > 0);
          inpos += count;
          ENSURE(inpos <= inbuf.size());
          if(inpos >= 4) {
            //log->info("Here!");
            char *dataPtr = &inbuf[0];
            uint32_t typeCode = be32toh(*(uint32_t*)dataPtr);
            dataPtr += 4;
            if(typeCode == 0x3 && inpos == inbuf.size()) {
              // sendRow
              ENSURE(be32toh(*(uint32_t*)dataPtr) == handle.id);
              dataPtr += 4;
              uint64_t rowIdx = htobe64(*(uint64_t*)dataPtr);
              dataPtr += 8;
              *reinterpret_cast<uint64_t*>(&outbuf[0]) = be64toh(numCols * 8);
              // treat the output as uint64_t[] instead of double[] to avoid type punning issues with be64toh
              //log->info("Starting writing row {} to out buffer", rowIdx);
              for(uint64_t colIdx = 0; colIdx < numCols; ++colIdx) {
                  rowbuf[colIdx] = *(matrix->LockedBuffer(matrix->LocalRow(rowIdx), matrix->LocalCol(colIdx)));
              }
              //log->info("Filled temporary buffer");
              auto invals = reinterpret_cast<const uint64_t*>(rowbuf.data());
              auto outvals = reinterpret_cast<uint64_t*>(&outbuf[8]);
              for(uint64_t colIdx = 0; colIdx < numCols; ++colIdx) {
                outvals[colIdx] = be64toh(invals[colIdx]);
              }
              //log->info("Finished writing row {} to out buffer", rowIdx);
              inpos = 0;
              pollEvents = POLLOUT; // after parsing the request, send the data
              break;
            }
          }
        }
      }
    }

    // handle writes
    if(revents & POLLOUT && pollEvents & POLLOUT) {
      // a la https://stackoverflow.com/questions/12170037/when-to-use-the-pollout-event-of-the-poll-c-function
      // and http://www.kegel.com/dkftpbench/nonblocking.html
      while(!isClosed()) {
        int count = write(sock, &outbuf[outpos], outbuf.size() - outpos);
        //std::cerr << format("%s: write: sock=%s, outbuf=%s, outpos=%s, count=%s\n")
        //    % world.rank() % sock % outbuf.size() % outpos % count;
        if (count == 0) {
          break;
        } else if(count == -1) {
          if(errno == EAGAIN) {
            // out buffer is full for now, wait for next POLLOUT
            break;
          } else if(errno == EINTR) {
            // interrupted (e.g. by signal), so try again
            continue;
          } else if(errno == ECONNRESET) {
            close();
            break;
          } else {
            // TODO
            abort();
          }
        } else {
          ENSURE(count > 0);
          outpos += count;
          ENSURE(outpos <= outbuf.size());
          if (outpos == outbuf.size()) { // after sending the row, wait for the next request
            rowsCompleted += 1;
            outpos = 0;
            pollEvents = POLLIN;
            break;
          }
        }
      }
    }

    return rowsCompleted;
  }
};

struct WorkerClientReceiveHandler {
  int sock;
  short pollEvents;
  std::vector<char> inbuf;
  size_t pos;
  DistMatrix *matrix;
  MatrixHandle handle;
  std::shared_ptr<spdlog::logger> log;

  WorkerClientReceiveHandler(int sock, std::shared_ptr<spdlog::logger> log, MatrixHandle handle, DistMatrix *matrix) :
      sock(sock), log(log), pollEvents(POLLIN), inbuf(matrix->Width() * 8 + 24),
      pos(0), matrix(matrix), handle(handle) {
  }

  ~WorkerClientReceiveHandler() {
    close();
  }

  bool isClosed() const {
    return sock == -1;
  }

  void close() {
    if(sock != -1) ::close(sock);
    //log->warn("Closed socket");
    sock = -1;
    pollEvents = 0;
  }

  int handleEvent(short revents) {
    mpi::communicator world;
    int rowsCompleted = 0;
    if(revents & POLLIN && pollEvents & POLLIN) {
      while(!isClosed()) {
        //log->info("waiting on socket");
        int count = recv(sock, &inbuf[pos], inbuf.size() - pos, 0);
        //log->info("count of received bytes {}", count);
        if(count == 0) {
          break;
        } else if(count == -1) {
          if(errno == EAGAIN) {
            // no more input available until next POLLIN
            //log->warn("EAGAIN encountered");
            break;
          } else if(errno == EINTR) {
            //log->warn("Connection interrupted");
            continue;
          } else if(errno == ECONNRESET) {
            //log->warn("Connection reset");
            close();
            break;
          } else {
            log->warn("Something else happened to the connection");
            // TODO
            abort();
          }
        } else {
          ENSURE(count > 0);
          pos += count;
          ENSURE(pos <= inbuf.size());
          if(pos >= 4) {
            char *dataPtr = &inbuf[0];
            uint32_t typeCode = be32toh(*(uint32_t*)dataPtr);
            dataPtr += 4;
            if(typeCode == 0x1 && pos == inbuf.size()) {
              // addRow
              size_t numCols = matrix->Width();
              ENSURE(be32toh(*(uint32_t*)dataPtr) == handle.id);
              dataPtr += 4;
              uint64_t rowIdx = htobe64(*(uint64_t*)dataPtr);
              dataPtr += 8;
              ENSURE(rowIdx < (size_t)matrix->Height());
              ENSURE(matrix->IsLocalRow(rowIdx));
              ENSURE(htobe64(*(uint64_t*)dataPtr) == numCols * 8);
              dataPtr += 8;
              auto localRowIdx = matrix->LocalRow(rowIdx);
              //log->info("Received row {} of matrix {}, writing to local row {}", rowIdx, handle.id, localRowIdx);
              for (size_t colIdx = 0; colIdx < numCols; ++colIdx) {
                double value = ntohd(*(uint64_t*)dataPtr);
                matrix->SetLocal(localRowIdx, matrix->LocalCol(colIdx), value); //LocalCal call should be unnecessary
                dataPtr += 8;
              }
              ENSURE(dataPtr == &inbuf[inbuf.size()]);
              //log->info("Successfully received row {} of matrix {}", rowIdx, handle.id);
              rowsCompleted++;
              pos = 0;
            } else if(typeCode == 0x2) {
              //log->info("All the rows coming to me from one Spark executor have been received");
              /**struct sockaddr_storage addr;
              socklen_t len;
              char peername[255];
              int result = getpeername(sock, (struct sockaddr*)&addr, &len);
              ENSURE(result == 0);
              getnameinfo((struct sockaddr*)&addr, len, peername, 255, NULL, 0, 0);
              log->info("Received {} rows from {}", rowsCompleted, peername);
              **/
              pos = 0;
            }
          }
        }
      }
    }
    //log->info("returning from handling events");
    return rowsCompleted;
  }
};

void Worker::sendMatrixRows(MatrixHandle handle, const El::AbstractDistMatrix<double> * matrix) {
  auto numRowsFromMe = matrix->LocalHeight();
  std::vector<std::unique_ptr<WorkerClientSendHandler>> clients;
  std::vector<pollfd> pfds;
  while(numRowsFromMe > 0) {
    pfds.clear();
    for(auto it = clients.begin(); it != clients.end();) {
      const auto &client = *it;
      if(client->isClosed()) {
        it = clients.erase(it);
      } else {
        pfds.push_back(pollfd{client->sock, client->pollEvents});
        it++;
      }
    }
    pfds.push_back(pollfd{listenSock, POLLIN}); // must be last entry
    int count = poll(&pfds[0], pfds.size(), -1); 
    if(count == -1 && (errno == EAGAIN || errno == EINTR)) continue;
    ENSURE(count != -1);
    //log->info("Monitoring {} sockets (one is the listening socket)", pfds.size());
    for(size_t idx=0; idx < pfds.size() && count > 0; ++idx) {
      auto curSock = pfds[idx].fd;
      auto revents = pfds[idx].revents;
      if(revents != 0) {
        count--;
        if(curSock == listenSock) {
          ENSURE(revents == POLLIN);
          sockaddr_in addr;
          socklen_t addrlen = sizeof(addr);
          int clientSock = accept(listenSock, reinterpret_cast<sockaddr*>(&addr), &addrlen);
          ENSURE(addrlen == sizeof(addr));
          ENSURE(fcntl(clientSock, F_SETFL, O_NONBLOCK) != -1);
          std::unique_ptr<WorkerClientSendHandler> client(new WorkerClientSendHandler(clientSock, log, handle, matrix, matrix->Width()));
          clients.push_back(std::move(client));
        } else {
          ENSURE(clients[idx]->sock == curSock);
          numRowsFromMe -= clients[idx]->handleEvent(revents);
        }
      }
    }
  }
  std::cerr << format("%s: finished sending rows\n") % world.rank();
}

void Worker::receiveMatrixBlocks(MatrixHandle handle) {
  std::vector<std::unique_ptr<WorkerClientReceiveHandler>> clients;
  std::vector<pollfd> pfds;
  uint64_t rowsLeft = matrices[handle].get()->LocalHeight(); 
  while(rowsLeft > 0) {
    //log->info("{} rows remaining", rowsLeft);
    pfds.clear();
    for(auto it = clients.begin(); it != clients.end();) {
      const auto &client = *it;
      if(client->isClosed()) {
        it = clients.erase(it);
      } else {
        pfds.push_back(pollfd{client->sock, client->pollEvents});
        it++;
      }
    }
    pfds.push_back(pollfd{listenSock, POLLIN});  // must be last entry
    //log->info("Pushed active clients to the polling list and added listening socket");
    int count = poll(&pfds[0], pfds.size(), -1);
    if(count == -1 && (errno == EAGAIN || errno == EINTR)) continue;
    ENSURE(count != -1);
    //log->info("Polled, now handling events");
    for(size_t idx = 0; idx < pfds.size() && count > 0; ++idx) {
      auto curSock = pfds[idx].fd;
      auto revents = pfds[idx].revents;
      if(revents != 0) {
        count--;
        if(curSock == listenSock) {
          ENSURE(revents == POLLIN);
          sockaddr_in addr;
          socklen_t addrlen = sizeof(addr);
          int clientSock = accept(listenSock, reinterpret_cast<sockaddr*>(&addr), &addrlen);
          ENSURE(addrlen == sizeof(addr));
          ENSURE(fcntl(clientSock, F_SETFL, O_NONBLOCK) != -1);
          std::unique_ptr<WorkerClientReceiveHandler> client(new WorkerClientReceiveHandler(clientSock, log, handle, matrices[handle].get()));
          clients.push_back(std::move(client));
          //log->info("Added new client");
        } else {
          ENSURE(clients[idx]->sock == curSock);
          //log->info("Handling a client's events");
          rowsLeft -= clients[idx]->handleEvent(revents);
        }
      }
    }
  }
}

int Worker::main() {
  // log to console as well as file (single-threaded logging)
  // TODO: allow to specify log directory, log level, etc.
  // TODO: make thread-safe
  // TODO: currently both stderr and logfile share the same report levels (can't have two sinks on same log with different level); use a split sink
  //  a la https://github.com/gabime/spdlog/issues/345 to allow stderr to print error messages only
  time_t rawtime = time(0);
  struct tm * timeinfo = localtime(&rawtime);
  char buf[80];
  strftime(buf, 80, "%F-%T", timeinfo);
  std::vector<spdlog::sink_ptr> sinks;
  auto stderr_sink = std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>(); // with ANSI color
  auto logfile_sink = std::make_shared<spdlog::sinks::simple_file_sink_st>(str(format("rank-%d-%s.log") % world.rank() % buf));

  stderr_sink->set_level(spdlog::level::err);
  logfile_sink->set_level(spdlog::level::info); // change to warn for production
  sinks.push_back(stderr_sink);
  sinks.push_back(logfile_sink);
  log = std::make_shared<spdlog::logger>( str(format("worker-%d") % world.rank()),
      std::begin(sinks), std::end(sinks));
  log->flush_on(spdlog::level::info);

  log->info("Started worker");
  log->info("Max number of OpenMP threads: {}", omp_get_max_threads());

  // create listening socket, bind to an available port, and get the port number
  ENSURE((listenSock = socket(AF_INET, SOCK_STREAM, 0)) != -1);
  sockaddr_in addr = {AF_INET};
  ENSURE(bind(listenSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
  ENSURE(listen(listenSock, 1024) == 0);
  ENSURE(fcntl(listenSock, F_SETFL, O_NONBLOCK) != -1);
  socklen_t addrlen = sizeof(addr);
  ENSURE(getsockname(listenSock, reinterpret_cast<sockaddr*>(&addr), &addrlen) == 0);
  ENSURE(addrlen == sizeof(addr));
  uint16_t port = be16toh(addr.sin_port);

  // transmit WorkerInfo to driver
  char hostname[256];
  ENSURE(gethostname(hostname, sizeof(hostname)) == 0);
  WorkerInfo info{hostname, port};
  world.send(0, 0, info);
  log->info("Listening for a connection at {}:{}", hostname, port);

  // handle commands until done
  while(!shouldExit) {
    const Command *cmd = nullptr;
    mpi::broadcast(world, cmd, 0);
    cmd->run(this);
    delete cmd;
  }

  // synchronized exit
  world.barrier();
  return EXIT_SUCCESS;
}

int workerMain(const mpi::communicator &world, const mpi::communicator &peers) {
  return Worker(world, peers).main();
}

} // namespace alchemist
