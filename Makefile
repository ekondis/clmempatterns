#OCLSDKDIR = /opt/cuda
#OCLSDKDIR = /usr/local/cuda
OCLSDKDIR = ${AMDAPPSDKROOT}
CXX = gcc
OCLSDKINC = ${OCLSDKDIR}/include
OCLSDKLIB = ${OCLSDKDIR}/lib/x86_64/
OPTFLAG = -O2
FLAGS = ${OPTFLAG} -I${OCLSDKINC} -std=c99
LFLAGS = ${OMPFLAG} ${PROFLAG} -L${OCLSDKLIB}
#LIBPARS = -lOpenCL -lrt

clmempatterns: main.o
	${CXX} ${LFLAGS} -o $@ $^ ${LIBPARS}

main.o: main.c
	${CXX} -c ${FLAGS} $<

clean:
	rm clmempatterns main.o
