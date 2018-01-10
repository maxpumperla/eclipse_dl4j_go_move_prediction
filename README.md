# Go move prediction with Eclipse Deeplearning4j

A little use case for Go move prediction with DL4J. This example reads in previously
created features and labels from resources, builds a convolutional neural network with
DL4J, trains and evaluates it.

## Getting started

To run this example, use maven as follows:

```{bash}
mvn clean install
MAVEN_OPTS='-Xmx6G' mvn exec:java
```

or use docker to do the same thing:

```{bash}
docker build . -t dl4jgo
docker run -e MAVEN_OPTS=-Xmx6G dl4jgo mvn "exec:java"
```
