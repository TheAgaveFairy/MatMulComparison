CC = gcc
CFLAGS = -g -fopenmp

all: cmatmul

cmatmul: cmatmul.c
	$(CC) $(CFLAGS) -o $@ $^
clean:
	rm -f main *.o
