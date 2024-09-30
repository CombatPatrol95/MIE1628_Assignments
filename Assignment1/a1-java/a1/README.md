### To build this image:
```shell
docker build -t mapreduce .
```

### To run this image as container with params:
```shell
docker run -v $pwd/out:/output mapreduce:latest [numberOfClusters] [maxIterations]
```