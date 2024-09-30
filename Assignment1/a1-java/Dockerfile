FROM openjdk:8-jdk-slim

# Install Hadoop
RUN apt-get update && apt-get install -y wget && \
    wget https://downloads.apache.org/hadoop/common/hadoop-3.4.0/hadoop-3.4.0.tar.gz && \
    tar -xzf hadoop-3.4.0.tar.gz && \
    mv hadoop-3.4.0 /opt/hadoop && \
    rm hadoop-3.4.0.tar.gz

# Set Hadoop environment variables
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

COPY data_points.txt /input/
COPY shakespeare.txt /input/


COPY a1/a1-q1/target/a1-q1-1.jar /app/linecount.jar
COPY a1/a1-q2/target/a1-q2-1.jar /app/kmeans.jar

# Run the MapReduce jobs
# K-means as 3 clusters and 10 iterations
ENTRYPOINT ["bash", "-c", \
            "hadoop jar /app/linecount.jar LineCountMapReduce /input/shakespeare.txt /output/output-linecount && \
             hadoop jar /app/kmeans.jar KMeansClusteringMR /input/datapoint.txt /output/output-kmeans $1 $2"]