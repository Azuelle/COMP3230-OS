// file: lab4-pthread.c
#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 4
#define NUM_ITERATIONS 1000000

int counter = 0;

// TODO3: define global variables (~1 line)
pthread_mutex_t mutex;

void *count_up(void *arg) {
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // TODO3: Protect the counter increment operation to prevent race
        // conditions
        pthread_mutex_lock(&mutex);
        counter++;
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    pthread_t threads[NUM_THREADS];

    // TODO3: init condition variables/semphore (~1 line)
    pthread_mutex_init(&mutex, NULL);

    for (int i = 0; i < NUM_THREADS; i++) {
        // TODO1: Create multiple threads to execute count_up (~1 line)
        pthread_create(&threads[i], NULL, count_up, NULL);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        // TODO2: wait all the threads to finish running (~1 line)
        pthread_join(threads[i], NULL);
    }

    // TODO3: free condition variables/semphore (~1 line)
    pthread_mutex_destroy(&mutex);

    printf("Final counter value: %d\n", counter);
    printf("Expected counter value: %d\n", NUM_ITERATIONS * NUM_THREADS);

    return 0;
}