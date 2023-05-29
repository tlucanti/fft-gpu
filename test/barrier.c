
#include <barrier.h>

inline void barrier_init(struct barrier *barrier, int total_threads)
{
	atomic_store(&barrier->waiting_threads, 0);
	atomic_store(&barrier->current_iteration, -1);
	barrier->total_threads = total_threads;
}


inline void barrier_wait(struct barrier *barrier, int id, int iter)
{
	atomic_fetch_add(&barrier->waiting_threads, 1);

	if (id == 0) {
		while (atomic_load(&barrier->waiting_threads) < barrier->total_threads) {
			sched_yield();
		}
		atomic_store(&barrier->waiting_threads, 0);
		atomic_fetch_add(&barrier->current_iteration, 1);
	} else {
		while (atomic_load(&barrier->current_iteration) < iter) {
			sched_yield();
		}
	}
}


//#define BARRIER_TEST

#ifdef BARRIER_TEST

struct barrier barrier;

void *routine(void *id)
{
	int pid = (unsigned long)id;

	for (int i = 0; i < 3; ++i) {
		printf("thread %d at iteration %d\n", pid, i);
		barrier_wait(&barrier, pid, i);
	}
	return NULL;
}

void test_barrier(void)
{
	const unsigned long thread_cnt = 4;
	pthread_t threads[thread_cnt];

	barrier_init(&barrier, thread_cnt);
	for (unsigned long i = 0; i < thread_cnt; ++i) {
		if (pthread_create(&threads[i], NULL, routine, (void *)i) != 0) {
			printf("pthread_create error\n");
			exit(1);
		}
	}
	for (int i = 0; i < thread_cnt; ++i) {
		if (pthread_join(threads[i], NULL) != 0) {
			printf("pthread_join error\n");
			exit(1);
		}
	}
}

int main() {
	test_barrier();
}

#endif
