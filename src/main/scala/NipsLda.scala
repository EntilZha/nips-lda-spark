package nipslda

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx.lib.LDA
import scala.io
import scala.collection.mutable.Map

object NipsLda {
  def edgesVocabFromText(sc:SparkContext):
                        (RDD[(LDA.WordId, LDA.DocId)], Array[String], Map[String, LDA.WordId]) = {
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
    (edges, vocab, vocabLookup)
  }
  def edgesVocabFromEdgeListDictionary(sc:SparkContext):
                                      (RDD[(LDA.WordId, LDA.DocId)], Array[String], Map[String, LDA.WordId]) = {
    val doc = sc.textFile("data/numeric-nips/counts.tsv")
    val edges = doc.flatMap(line => {
      val l = line.split("\t")
      val wordId:LDA.WordId = l(0).toLong
      val docId:LDA.DocId = l(1).toLong
      val occurrences = l(2).toInt
      List.fill[(LDA.WordId, LDA.DocId)](occurrences)((wordId, docId))
    })
    val vocab = io.Source.fromFile("~/nips-lda-spark/data/numeric-nips/dictionary.txt").getLines().toArray
    var vocabLookup = scala.collection.mutable.Map[String, LDA.WordId]()
    for (i <- 0 until vocab.length) {
      vocabLookup += vocab(i) -> i
    }
    (edges, vocab, vocabLookup)
  }
  def main(args:Array[String]): Unit = {
    val serializer = "org.apache.spark.serializer.KryoSerializer"
    val conf = new SparkConf()
                  .setMaster("spark://ec2-54-213-199-91.us-west-2.compute.amazonaws.com:7077")
                  .setAppName("nips-lda")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryo.registrator", "org.apache.spark.graphx.GraphKryoRegistrator")
    val sc = new SparkContext(conf)
    sc.addSparkListener(new org.apache.spark.scheduler.JobLogger())
    val (edges, vocab, vocabLookup) = edgesVocabFromEdgeListDictionary(sc)
    val model = new LDA(edges, 50, loggingInterval = 1, loggingLikelihood = false, loggingTime = false)
    val ITERATIONS = 10
    model.train(ITERATIONS)
    val words = model.topWords(15)
    sc.stop()
    for (i <- 0 until 50) {
      print("Topic " + i.toString + ": ")
      for (w <- 0 until 15) {
        val word = vocab(words(i)(w)._2.toInt)
        val count = words(i)(w)._1
        print(count.toString + "*" + word + " ")
      }
      println()
    }
  }
}