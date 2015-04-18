#OCLSDKDIR = /opt/cuda
#OCLSDKDIR = /usr/local/cuda
OCLSDKDIR = ${AMDAPPSDKROOT}
CXX = gcc
OCLSDKINC = ${OCLSDKDIR}/include
OCLSDKLIB = ${OCLSDKDIR}/lib/x86_64/
OPTFLAG = -O2
FLAGS = ${OPTFLAG} -I${OCLSDKINC} -std=c99
LFLAGS = ${OMPFLAG} ${PROFLAG} -L${OCLSDKLIB}
LIBPARS = -lOpenCL -lrt

clmempatterns: main.o
	${CXX} ${LFLAGS} -o $@ $^ ${LIBPARS}

main.o: main.c kernel.h
	${CXX} -c ${FLAGS} $<

kernel.h: kernel.cl
	echo "const char c_kernel[]={" >kernel.h
	hexdump -ve '1/1 "0x%.2x,"' kernel.cl >>kernel.h
	echo "0x00};" >>kernel.h

clean:
	rm clmempatterns main.o kernel.h
