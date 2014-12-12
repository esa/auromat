# Graph the dependencies for auromat.

ROOT=..

PDFS =						\
	auromat.pdf auromat_nodeps.pdf auromat_nodeps_grouped.pdf

include Makefile.rules


# Handmade cluster and filter lists.
auromat.clusters: clusters
	cp $< $@
	
auromat.filters: filters_stdlib filters_auromat
	cat $^ > $@

auromat_nodeps.clusters: clusters
	cp $< $@

auromat_nodeps.filters: filters_stdlib filters_auromat filters_deps filters_auromat_nodeps
	cat $^ > $@
	
auromat_nodeps_grouped.clusters: clusters_grouped
	cp $< $@

auromat_nodeps_grouped.filters: filters_stdlib filters_auromat filters_deps filters_auromat_nodeps
	cat $^ > $@