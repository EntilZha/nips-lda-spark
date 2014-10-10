package nipslda

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx.algorithms.LDA

object Main {
  def main(args:Array[String]): Unit = {
    val serializer = "org.apache.spark.serializer.KryoSerializer"
    val conf = new SparkConf()
                  .setMaster("local")
                  .setAppName("nips-lda")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryo.registrator", "org.apache.spark.graphx.GraphKryoRegistrator")
    val sc = new SparkContext(conf)
    val docs = sc.wholeTextFiles("src/main/resources/nipstxt/**")
    val tokens = docs.flatMap({case (name, contents) => contents.split(' ')})
    val (vocab, vocabLookup) = LDA.extractVocab(tokens)
    val edges = docs.flatMap({case (name, contents) =>
      contents.split(' ').map(token => (vocabLookup(token), Math.abs(name.hashCode).asInstanceOf[LDA.DocId]))
    })
    val model = new LDA(edges)
    model.iterate(1)
    println(model.topWords(5))
    sc.stop()
  }
}