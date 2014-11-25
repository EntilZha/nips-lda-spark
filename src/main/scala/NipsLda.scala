package nipslda

import org.apache.spark.mllib.util.LDADataGenerator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx.lib.LDA
import scala.io
import scala.collection.mutable.Map
import breeze.stats.distributions

object NipsLda {
  def edgesVocabFromText(sc:SparkContext):
                        (RDD[(LDA.WordId, LDA.DocId)], Array[String], Map[String, LDA.WordId]) = {
    val stopWords = io.Source.fromFile("/root/nips-lda-spark/data/stop-words.txt").getLines().map(l => l.trim()).toSet
    val docs = sc.textFile("s3n://files.sparks.public/data/enwiki_category_text/part-0000[0-1]").map({contents =>
      (contents.hashCode(), contents.replaceAll("[^A-Za-z']+", " ").trim.toLowerCase.split("\\s+").filter(w => !stopWords(w)))
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
    val doc = sc.textFile("s3n://amplab-lda/counts.tsv")
    //val doc = sc.textFile("/Users/pedro/Code/nips-lda/data/numeric-nips/counts.tsv")
    val edges = doc.flatMap(line => {
      val l = line.split("\t")
      val wordId:LDA.WordId = l(0).toLong
      val docId:LDA.DocId = l(1).toLong
      val occurrences = l(2).toInt
      List.fill[(LDA.WordId, LDA.DocId)](occurrences)((wordId, docId))
    })
    val vocab = io.Source.fromFile("/root/nips-lda-spark/data/numeric-nips/dictionary.txt").getLines().toArray
    //val vocab = io.Source.fromFile("/Users/pedro/Code/nips-lda/data/numeric-nips/dictionary.txt").getLines().toArray
    var vocabLookup = scala.collection.mutable.Map[String, LDA.WordId]()
    for (i <- 0 until vocab.length) {
      vocabLookup += vocab(i) -> i
    }
    (edges, vocab, vocabLookup)
  }

  def runGenerativeLDA(sc:SparkContext): Unit = {
    val alpha = 0.1
    val beta = 0.1
    val nTopics = 100
    val nDocs = 2483
    val nWords = 14036
    val nTokensPerDoc = 1321
    val corpus = LDADataGenerator.generateCorpus(sc, alpha, beta, nTopics, nDocs, nWords, nTokensPerDoc)
    val model = new LDA(corpus, loggingInterval = 1, loggingTime = true)
    val iterations = 10
    model.train(iterations)
    sc.stop()
  }

  def getSparkContext(master:String):SparkContext = {
    val conf = new SparkConf()
                  .setMaster(master)
                  .setAppName("LDA")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryo.registrator", "org.apache.spark.graphx.GraphKryoRegistrator")
    return new SparkContext(conf)
  }

  def runLDA(sc:SparkContext): Unit = {
    println(sc.getConf.getAll.mkString(","))
    sc.addSparkListener(new org.apache.spark.scheduler.JobLogger())
    val (edges, vocab, vocabLookup) = edgesVocabFromText(sc)
    val NUM_TOPICS = 100
    val model = new LDA(edges, NUM_TOPICS, loggingInterval = 1, loggingLikelihood = false, loggingTime = true)
    val ITERATIONS = 10
    model.train(ITERATIONS)
    val words = model.topWords(15)
    sc.stop()
    for (i <- 0 until NUM_TOPICS) {
      print("Topic " + i.toString + ": ")
      for (w <- 0 until 15) {
        val word = vocab(words(i)(w)._2.toInt)
        val count = words(i)(w)._1
        print(count.toString + "*" + word + " ")
      }
      println()
    }
  }

  def main(args:Array[String]): Unit = {
    if (args.length == 0) {
      throw new Exception("Must specify master")
    }
    if (args.length == 1 || args(1) == "lda") {
      runLDA(getSparkContext(args(0)))
    }
    if (args(1) == "gen") {
      runGenerativeLDA(getSparkContext(args(0)))
    }
  }
}
