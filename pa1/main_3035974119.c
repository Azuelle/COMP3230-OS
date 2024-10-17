/*
 * PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
 * FILE NAME: main_3035974119.c
 * NAME: TANG Jiakai
 * UID: 3035974119
 * Development Platform: Ubuntu 22.04.4 LTS aarch64
 *                       (in Docker container on macOS)
 * Remark: (How much you implemented?)
 * How to compile separately: gcc -o main main_3035974119.c
 */

#include "common.h"  // common definitions

#include <stdio.h>    // for printf, fgets, scanf, perror
#include <stdlib.h>   // for exit() related
#include <unistd.h>   // for folk, exec...
#include <wait.h>     // for waitpid
#include <signal.h>   // for signal handlers and kill
#include <string.h>   // for string related
#include <sched.h>    // for sched-related
#include <syscall.h>  // for syscall interface

#define READ_END 0      // helper macro to make pipe end clear
#define WRITE_END 1     // helper macro to make pipe end clear
#define SYSCALL_FLAG 0  // flags used in syscall, set it to default 0

int inference_pid = -1;

// Kill inference process when SIGINT is received
void handle_sigint(int signum) {
    kill(inference_pid, SIGINT);
    int status;
    waitpid(inference_pid, &status, 0);
    printf("Inference process exited with code %d.\n", status);
    exit(EXIT_SUCCESS);
}

// Start new prompt input when SIGUSR1 is received
void handle_sigusr1(int signum) {}

int main(int argc, char* argv[]) {
    char* seed;  //
    if (argc == 2) {
        seed = argv[1];
    } else if (argc == 1) {
        // use 42, the answer to life the universe and everything, as default
        seed = "42";
    } else {
        fprintf(stderr, "Usage: ./main <seed>\n");
        fprintf(stderr, "Note:  default seed is 42\n");
        exit(1);
    }

    // Create pipe
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        perror("pipe failed");
        exit(EXIT_FAILURE);
    }

    // Fork and exec inference process
    inference_pid = fork();
    if (inference_pid < 0) {
        perror("fork failed");
        exit(EXIT_FAILURE);
    } else if (inference_pid == 0) {
        // CHILD PROCESS

        // Redirect to stdin
        close(pipefd[WRITE_END]);
        if (dup2(pipefd[READ_END], STDIN_FILENO) == -1) {
            perror("dup2 failed");
            exit(EXIT_FAILURE);
        }

        // Exec inference process
        char* args[] = {"./inference", seed, NULL};
        execvp(args[0], args);
        perror("execvp failed");
        exit(EXIT_FAILURE);
    }
    // PARENT PROCESS

    // Close unused pipe end
    close(pipefd[READ_END]);

    // Define signal handlers
    signal(SIGINT, handle_sigint);
    signal(SIGUSR1, handle_sigusr1);

    // Main loop
    while (1) {
    }

    // Clean up
    close(pipefd[WRITE_END]);

    return EXIT_SUCCESS;
}