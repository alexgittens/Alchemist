package amplab.alchemist
// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.sql.Row
// spark-sql
import org.apache.spark.sql.SparkSession

object ConvertH5ToParquet {
    def main(args: Array[String]): Unit = {
        val spark = (SparkSession
                      .builder()
                      .appName("ConvertH5ToParquet")
                      .getOrCreate())
        val sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        import spark.implicits._
        
        val al = new Alchemist(sc)

        val infname : String = args(0).toString
        val outfname : String = args(1).toString
        val varname : String = args(2).toString

        var t1 = System.nanoTime()
        val alDset = al.readHDF5(infname, varname)
        var t2 = System.nanoTime()
        println(" ")
        println(s"Finished loading dataset in Alchemist, took ${(t2-t1) * 1.0E-9} seconds")

        var t3 = System.nanoTime()
        val A = alDset.getIndexedRowMatrix()
        var t4 = System.nanoTime()
        println(" ")
        println(s"Retrieved dataset to Spark, took ${(t4 - t3) * 1.0E-9} seconds")

        var t5 = System.nanoTime()
        A.rows.map( x => (x.index, x.vector)).toDF("index", "vector").write.parquet(outfname)
        var t6 = System.nanoTime()
        println(" ")
        println(s"Finished writing out to Parquet, took ${(t6 - t5) * 1.0E-9} seconds")

        al.stop
        sc.stop
    }
}
