# Makefile for mpi/torchrun plugin

AUTOMAKE_OPTIONS = foreign

PLUGIN_FLAGS = -module -avoid-version --export-dynamic

AM_CPPFLAGS = -DSLURM_PLUGIN_DEBUG -I$(top_srcdir) -I$(top_srcdir)/src/common

pkglib_LTLIBRARIES = mpi_torchrun.la

# Torchrun MPI plugin
mpi_torchrun_la_SOURCES = mpi_torchrun.c
mpi_torchrun_la_CFLAGS = $(AM_CFLAGS)
mpi_torchrun_la_LDFLAGS = $(PLUGIN_FLAGS)
