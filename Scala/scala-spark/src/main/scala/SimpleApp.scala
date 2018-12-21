import org.apache.spark.{SparkConf, SparkContext}

object SimpleApp {

  def main(args: Array[String]): Unit = {
    val logFile = "README.md"

    val conf = new SparkConf().setAppName("Simplae App").setMaster("local")

    val sc = new SparkContext(conf)

    val logData = sc.textFile(logFile, 2).cache()


    val lineCount = logData.count()
    val lineCollect = logData.collect()

    println(lineCollect)
    println("for test", lineCount)
    sc.stop()

  }

}
