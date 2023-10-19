all:
	hipcc -O3 -lnuma -lpthread -D__HIP_PLATFORM_AMD__ hits.c -o hits

debug:
	hipcc -Wall -g -lnuma -lpthread -D__HIP_PLATFORM_AMD__ hits.c -o hits

clean:
	@rm -f hits

