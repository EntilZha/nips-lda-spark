package nipslda

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx.lib.LDA
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
    val stopWords = io.Source.fromFile("data/stop-words.txt").getLines().map(l => l.trim()).toSet
    val docs = sc.wholeTextFiles("data/nipstxt/**").map({case (name, contents) =>
      (name, contents.replaceAll("[^A-Za-z']+", " ").trim.toLowerCase.split("\\s+").filter(w => !stopWords(w)))
    })
    val tokens = docs.flatMap({case (name, contents) => contents})
    val (vocab, vocabLookup) = LDA.extractVocab(tokens)
    val edges = docs.flatMap({case (name, contents) =>
      contents.map(
        token => (vocabLookup(token), Math.abs(name.hashCode).asInstanceOf[LDA.DocId])
      )
    })
    val model = new LDA(edges, 50)
    model.iterate(10)
    val words = model.topWords(15)
    sc.stop()
    for (i <- 0 to 49) {
      print("Topic " + i.toString + ": ")
      for (w <- 0 to 14) {
        val word = vocab(words(i)(w)._2.toInt)
        val count = words(i)(w)._1
        print(count.toString + "*" + word + " ")
      }
      println()
    }
  }
}