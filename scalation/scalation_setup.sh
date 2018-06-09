#!/bin/bash

cd /Applications

# install sbt (scala)
sudo port install sbt

# install scalation
wget http://cobweb.cs.uga.edu/~jam/scalation_1.4.tar.gz
tar xvfz scalation_1.4.tar.gz

# create and enter project directory
mkdir ~/Documents/HelloWorldOfDS
cd ~/Documents/HelloWorldOfDS

# source for automated build: Scala Cookbook Ch. 18.1
# generate file structure
mkdir -p src/{main,test}/{java,resources,scala}
mkdir lib project target
# create an initial build.sbt file
echo 'name := "HelloWorldOfDS"
version := "1.0"
scalaVersion := "2.12.4"' > build.sbt

# move necessary .jar files into project directory
scp -r /Applications/scalation_1.4/scalation_modeling/lib/ ./lib/
