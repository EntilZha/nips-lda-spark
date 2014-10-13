package nipslda

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx.algorithms.LDA
import scala.io

object Main {
  def main(args:Array[String]): Unit = {
    val serializer = "org.apache.spark.serializer.KryoSerializer"
    val conf = new SparkConf()
                  .setMaster("local")
                  .setAppName("nips-lda")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryo.registrator", "org.apache.spark.graphx.GraphKryoRegistrator")
    val sc = new SparkContext(conf)
    val docs = sc.wholeTextFiles("data/nipstxt/**")
    val stopWords = io.Source.fromFile("data/stop-words.txt").getLines().map(l => l.trim()).toSet
    val tokens = docs.flatMap({case (name, contents) => contents.split(' ').map(w => w.trim().toLowerCase)})
    assert(stopWords.contains("the"))
    val (vocab, vocabLookup) = LDA.extractVocab(tokens)
    for (e <- stopWords) {
      println(e)
    }
    val regex = "[A-Za-z]+".r
    val edges = docs.flatMap({case (name, contents) =>
      contents.split(' ').map(w => w.trim().toLowerCase).filter(w => !stopWords.contains(w) && regex.pattern.matcher(w).matches).map(
        token => (vocabLookup(token.toLowerCase), Math.abs(name.hashCode).asInstanceOf[LDA.DocId])
      )
    })
    val model = new LDA(edges, 10)
    model.iterate(10)
    val words = model.topWords(5)
    sc.stop()
    for (i <- 0 to 9) {
      print("Topic " + i.toString + ": ")
      for (w <- 0 to 4) {
        val word = vocab(words(i)(w)._2.toInt)
        val count = words(i)(w)._1
        print(count.toString + "*" + word + " ")
      }
      println()
    }
  }
}