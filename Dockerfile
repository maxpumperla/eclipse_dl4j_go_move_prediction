FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

ENV JAVA_PACKAGE_NAME=jdk-8u131-linux-x64.tar.gz \
    JAVA_ARCH_DOWNLOAD_URL=http://download.oracle.com/otn-pub/java/jdk/8u131-b11/d54c1d3a095b4ff2b6607d096fa80163 \
	JAVA_HOME=/opt/java \
	PATH=/opt/sbt/bin:/opt/cmake/bin:${PATH} \
	MAVEN_VERSION=3.3.9

RUN apt-get update && \
    apt-get -y --no-install-recommends install wget curl ca-certificates software-properties-common git && \
	add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
	apt-get update && \
	apt-get install -y --no-install-recommends gcc-4.9 g++-4.9 && \
	update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 50 && \
	update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 50 && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir -p /opt && \
    # Switch from fecth jdk with curl to wget, because of errors during the fetch
    wget --no-cookies --no-check-certificate \
        --header "Cookie: gpw_e24=http%3A%2F%2Fwww.oracle.com%2F; oraclelicense=accept-securebackup-cookie" \
        "${JAVA_ARCH_DOWNLOAD_URL}/${JAVA_PACKAGE_NAME}" && \
    tar -xzf "${JAVA_PACKAGE_NAME}" -C /opt && \
    rm "${JAVA_PACKAGE_NAME}" && \
	mv /opt/jdk* /opt/java && \
	update-alternatives --install /usr/bin/java java /opt/java/bin/java 100 && \
	update-alternatives --install /usr/bin/javac javac /opt/java/bin/javac 100

RUN mkdir -p /opt/maven && \
	curl -fsSL http://apache.osuosl.org/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz \
    | tar -xzC /opt/maven --strip-components=1 && \
	ln -s /opt/maven/bin/mvn /usr/bin/mvn

RUN mkdir -p /opt/sbt && \
	curl -fsSL https://dl.bintray.com/sbt/native-packages/sbt/0.13.13/sbt-0.13.13.tgz \
	| tar -xzC /opt/sbt --strip-components=1

RUN mkdir -p /opt/cmake && \
    curl -fsSL https://cmake.org/files/v3.9/cmake-3.9.0-Linux-x86_64.tar.gz \
    | tar -xzC /opt/cmake --strip-components=1

# Copy application to container
RUN mkdir -p app
WORKDIR /app
COPY . /app

RUN mvn clean install
RUN mvn exec:java"

RUN export MAVEN_OPTS=-Xmx6G

ENTRYPOINT ["mvn", "\"exec:java\""]
