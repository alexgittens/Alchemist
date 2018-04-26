package amplab.alchemist
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import scala.math.max
import java.nio.ByteBuffer

class AlMatrix(val al: Alchemist, val handle: MatrixHandle) {
  def getDimensions() : Tuple2[Long, Int] = {
    return al.client.getMatrixDimensions(handle)
  }

  def transpose() : AlMatrix = {
    new AlMatrix(al, al.client.getTranspose(handle))
  }

  // Caches result by default, because may not want to recreate (e.g. if delete referenced matrix on Alchemist side to save memory)
  def getIndexedRowMatrix() : IndexedRowMatrix = {
    val (numRows, numCols) = getDimensions()

    // Alchemist workers indicate which rows they have, the rows are randomly assigned to Spark workers, and they retrieve them from the relevant workers
    // Ignores non participating workers
    val workerRowIndices : Array[Tuple2[WorkerId, Array[Long]]] = al.client.getWorkerRowIndices(this.handle)
    val numPartitions = max(al.sc.defaultParallelism, al.client.workerCount)

    val layout : RDD[Tuple2[WorkerId, Long]] = al.sc.parallelize(
      workerRowIndices.flatMap{ case (workerid, rowIndices) => Array.fill(rowIndices.length)(workerid) zip rowIndices}, 
      numPartitions)

    /*
    println("Here're the (worker, rowIndex) tuples")
    println(
      layout.collect().map{ case(workerid, rowIndex) => s"(${workerid.id}, ${rowIndex})" }.mkString(" ")
    )
    */

    // capture references needed by the closure without capturing `this.al`
    val ctx = al.context
    val handle = this.handle

    al.client.getIndexedRowMatrixStart(handle)
    val rows : RDD[IndexedRow] = layout.mapPartitions( workerRowTuplesIterator => {
      val workerRowTuples = workerRowTuplesIterator.toArray
      val uniqueWorkers : Array[WorkerId] = workerRowTuples.map{ case(workerid, rowIndex) => workerid}.distinct

      val workerIterators : Seq[Iterator[IndexedRow]] = 
        uniqueWorkers.map{ curWorker =>
          val curRowIndices = workerRowTuples.toArray.filter{ case (workerId, rowIndex) => workerId.id == curWorker.id}.map(_._2)
          /*
          println(s"Asking worker ${curWorker.id} for rows:")
          println(curRowIndices.map(idx => idx.toString).mkString(" "))
          */
          val worker = ctx.connectWorker(curWorker)
          val result = curRowIndices.toList.map { rowIndex => 
            new IndexedRow(rowIndex, worker.getIndexedRowMatrix_getRow(handle, rowIndex, numCols))
          }.iterator
          worker.close()
          result
        }

      workerIterators.foldLeft(Iterator[IndexedRow]())(_ ++ _)
    }, preservesPartitioning=true)

    val result = new IndexedRowMatrix(rows, numRows, numCols)
    result.rows.cache()
    result.rows.count
    al.client.getIndexedRowMatrixFinish(handle)
    result
  }
}

  /**
    val rows = sacrificialRDD.mapPartitionsWithIndex( (idx, rowindices) => {
      var result : Iterator[IndexedRow] = Iterator.empty // it's possible not every partition will have data in it when the matrix being returned is small
      if (!rowindices.toList.isEmpty) {
          val worker = ctx.connectWorker(layout(idx))
          result = rowindices.toList.map { rowIndex =>
            new IndexedRow(rowIndex, worker.getIndexedRowMatrix_getRow(handle, rowIndex, numCols))
          }.iterator
          worker.close()
      } else {
        println(s"No rows were assigned to partition ${idx}, so returning an empty partition")
      }
      result
    }, preservesPartitioning=true)
    val result = new IndexedRowMatrix(rows, numRows, numCols)
    result.rows.cache()
    result.rows.count
    al.client.getIndexedRowMatrixFinish(handle)
    result
  }
  **/

object AlMatrix {
  def apply(al: Alchemist, mat: IndexedRowMatrix): AlMatrix = {
    // Sends the IndexedRowMatrix over to Alchemist as MD, STAR Elemental matrix
    
    val ctx = al.context
    val workerIds = ctx.workerIds
    // rowWorkerAssignments is an array of WorkerIds whose ith entry is the world rank of the alchemist worker
    // that will take the ith row (ranging from 0 to numworkers-1). Note 0 is an executor, not the driver
    val (handle, rowWorkerAssignments) = al.client.newMatrixStart(mat.numRows, mat.numCols)
    mat.rows.mapPartitionsWithIndex { (idx, part) =>
      val rows = part.toArray
      val relevantWorkers = rows.map(row => rowWorkerAssignments(row.index.toInt).id).distinct.map(id => new WorkerId(id))
      println("Sending data to following workers: ")
      println(relevantWorkers.map(node => node.id.toString).mkString(" "))
      val maxWorkerId = relevantWorkers.map(node => node.id).max
      var nodeClients = Array.fill(maxWorkerId+1)(None: Option[WorkerClient])
      System.err.println(s"Connecting to ${relevantWorkers.length} workers")
      relevantWorkers.foreach(node => nodeClients(node.id) = Some(ctx.connectWorker(node)))
      System.err.println(s"Successfully connected to all workers; have ${rows.length} rows to send")

      // TODO: randomize the order the rows are sent in to avoid queuing issues?
      var count = 0
      val buflen = 4 + 4 + 8 + 8 + 8 * rows(0).vector.toArray.length
      val reuseBuf = ByteBuffer.allocateDirect(Math.min(buflen, 16*1024*1024))
      rows.foreach{ row =>
        count += 1
//        System.err.println(s"Sending row ${row.index.toInt}, ${count} of ${rows.length}")
        nodeClients(rowWorkerAssignments(row.index.toInt).id).get.
          newMatrix_addRow(handle, row.index, row.vector.toArray, reuseBuf)
      }
      System.err.println("Finished sending rows")
      nodeClients.foreach(client => 
          if (client.isDefined) {
            client.get.newMatrix_partitionComplete(handle)
            client.get.close()
          })
      Iterator.single(true)
    }.count
    al.client.newMatrixFinish(handle)
    new AlMatrix(al, handle)
  }
}
