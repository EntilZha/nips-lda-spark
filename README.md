nips-lda-spark
==============
This repository is being used to test the LDA implementation based in [github.com/entilzha/spark](https://github.com/EntilZha/spark/blob/LDA/graphx/src/main/scala/org/apache/spark/graphx/lib/TopicModeling.scala)

To run this code, you will need:

1. The fork of spark above
2. Run ```sbt/sbt publish-local``` in the spark directory.
3. ```sbt/sbt run``` in this projects directory

To configure how the program runs, modify the parameters to LDA within the NipsLda source file.
