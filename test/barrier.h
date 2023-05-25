
#ifndef BARRIER_H
#define BARRIER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <pthread.h>

struct barrier {
	volatile atomic_int waiting_threads;
	volatile atomic_int current_iteration;
	int total_threads;
};

void barrier_init(struct barrier *barrier, int total_threads);
void barrier_wait(struct barrier *barrier, int id, int iter);

#endif /* BARRIER_H */

